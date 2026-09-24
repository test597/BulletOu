// Included inside the backend namespace after buffer validation helpers.
// One channel/bucket per CTA. Reference-first implementation: no atomics,
// double-precision moments, and exact BN backward (not a straight-through op).
// Layout: params=[gamma,beta], running=[mean,variance,initialized],
// stats=[inverse_std,count_used_for_batch_statistics].
__global__ void bn_ft_activate(float* a,float* b,float* combined,size_t rows,size_t width) {
    size_t j=blockIdx.x*blockDim.x+threadIdx.x,half=width/2;
    if(j>=rows*half)return;
    size_t base=(j/half)*width,u=j%half;
    float a0=crelu(a[base+u]),a1=crelu(a[base+half+u]);
    float b0=crelu(b[base+u]),b1=crelu(b[base+half+u]);
    a[base+u]=a0;a[base+half+u]=a1;b[base+u]=b0;b[base+half+u]=b1;
    combined[base+u]=a0*a1*SFNN_PAIRWISE_SCALE;
    combined[base+half+u]=b0*b1*SFNN_PAIRWISE_SCALE;
}
__global__ void bn_activate(float* a,size_t n) {
    size_t j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n)a[j]=crelu(a[j]);
}
__global__ void bn_bias_sum(const float* a,const float* b,float* gb,size_t rows,size_t width) {
    size_t u=blockIdx.x*blockDim.x+threadIdx.x;if(u>=width)return;
    float s=0;for(size_t i=0;i<rows;++i)s+=a[i*width+u]+b[i*width+u];gb[u]+=s;
}
__global__ void bn_forward_kernel(float* a, float* b, float* xa, float* xb,
    const int* buckets, const float* params, float* running, float* stats,
    size_t rows, size_t stride, size_t width, size_t groups,
    float epsilon, float momentum, bool training) {
    const size_t channel=blockIdx.x, unit=channel%width, group=channel/width;
    const size_t channels=groups*width;
    __shared__ double sums[256], counts[256];
    __shared__ double mean, variance, count;
    double s=0.0,n=0.0;
    for(size_t i=threadIdx.x;i<rows;i+=256) {
        if(groups>1 && buckets[i]!=static_cast<int>(group)) continue;
        s+=double(a[i*stride+unit]); n+=1.0;
        if(b) {s+=double(b[i*stride+unit]); n+=1.0;}
    }
    sums[threadIdx.x]=s; counts[threadIdx.x]=n;
    __syncthreads();
    for(int d=128;d;d/=2) {
        if(threadIdx.x<d) {sums[threadIdx.x]+=sums[threadIdx.x+d]; counts[threadIdx.x]+=counts[threadIdx.x+d];}
        __syncthreads();
    }
    if(threadIdx.x==0) {
        count=counts[0]; mean=count>0?sums[0]/count:0.0;
    }
    __syncthreads();
    s=0.0;
    if(training && count>=2.0) {
        for(size_t i=threadIdx.x;i<rows;i+=256) {
            if(groups>1 && buckets[i]!=static_cast<int>(group)) continue;
            double v=double(a[i*stride+unit])-mean; s+=v*v;
            if(b) {v=double(b[i*stride+unit])-mean; s+=v*v;}
        }
    }
    sums[threadIdx.x]=s;
    __syncthreads();
    for(int d=128;d;d/=2) {
        if(threadIdx.x<d) sums[threadIdx.x]+=sums[threadIdx.x+d];
        __syncthreads();
    }
    if(threadIdx.x==0) {
        if(training && count>=2.0) {
            variance=sums[0]/count;
            const float m=running[2*channels+channel]>0?momentum:1.0f;
            running[channel]=(1-m)*running[channel]+m*float(mean);
            // Unbiased running variance; biased batch variance in forward.
            running[channels+channel]=(1-m)*running[channels+channel]+m*float(sums[0]/(count-1.0));
            running[2*channels+channel]=1.0f;
            stats[channels+channel]=float(count);
        } else {
            mean=running[channel]; variance=running[channels+channel];
            stats[channels+channel]=0.0f;
        }
        stats[channel]=1.0f/sqrtf(float(variance)+epsilon);
    }
    __syncthreads();
    const float inv=stats[channel], gamma=params[channel], beta=params[channels+channel];
    for(size_t i=threadIdx.x;i<rows;i+=256) {
        if(groups>1 && buckets[i]!=static_cast<int>(group)) continue;
        size_t j=i*stride+unit;
        float x=float(double(a[j])-mean)*inv; xa[j]=x; a[j]=gamma*x+beta;
        if(b) {x=float(double(b[j])-mean)*inv; xb[j]=x; b[j]=gamma*x+beta;}
    }
}

__global__ void bn_backward_kernel(float* da,float* db,const float* xa,const float* xb,
    const int* buckets,const float* params,const float* stats,float* grads,
    size_t rows,size_t stride,size_t width,size_t groups) {
    const size_t ch=blockIdx.x,u=ch%width,g=ch/width,c=groups*width;
    __shared__ double sums[256],products[256];
    double s=0.0,p=0.0;
    for(size_t i=threadIdx.x;i<rows;i+=256) {
        if(groups>1 && buckets[i]!=static_cast<int>(g)) continue;
        const size_t j=i*stride+u;
        s+=da[j]; p+=double(da[j])*xa[j];
        if(db) {s+=db[j]; p+=double(db[j])*xb[j];}
    }
    sums[threadIdx.x]=s; products[threadIdx.x]=p;
    __syncthreads();
    for(int d=128;d;d/=2) {
        if(threadIdx.x<d) {sums[threadIdx.x]+=sums[threadIdx.x+d];products[threadIdx.x]+=products[threadIdx.x+d];}
        __syncthreads();
    }
    if(threadIdx.x==0) {grads[ch]+=float(products[0]);grads[c+ch]+=float(sums[0]);}
    const float n=stats[c+ch],scale=params[ch]*stats[ch];
    const float sm=n>0?float(sums[0])/n:0.0f,pm=n>0?float(products[0])/n:0.0f;
    for(size_t i=threadIdx.x;i<rows;i+=256) {
        if(groups>1 && buckets[i]!=static_cast<int>(g)) continue;
        const size_t j=i*stride+u;
        da[j]=scale*(da[j]-sm-xa[j]*pm);
        if(db) db[j]=scale*(db[j]-sm-xb[j]*pm);
    }
}

extern "C" int bulletou_bn_forward(BulletOuCudaCppContext* ctx,
    BulletOuCudaCppF32Buffer* a,BulletOuCudaCppF32Buffer* b,
    BulletOuCudaCppF32Buffer* xa,BulletOuCudaCppF32Buffer* xb,
    BulletOuCudaCppI32Buffer* ids,BulletOuCudaCppF32Buffer* params,
    BulletOuCudaCppF32Buffer* running,BulletOuCudaCppF32Buffer* stats,
    size_t rows,size_t stride,size_t width,size_t groups,float epsilon,float momentum,int training) {
    if(!rows || !width || stride<width || !groups || groups>65536/width || rows>SIZE_MAX/stride ||
       !std::isfinite(epsilon) || epsilon<=0 || !std::isfinite(momentum) || momentum<=0 || momentum>1)
        return fail_message("invalid batch normalization shape/configuration");
    const size_t n=rows*stride,c=groups*width;
    if(validate_buffer(ctx,a,n,"BN input") || validate_buffer(ctx,xa,n,"BN normalized input") ||
       validate_buffer(ctx,params,2*c,"BN affine") || validate_buffer(ctx,running,3*c,"BN running stats") ||
       validate_buffer(ctx,stats,2*c,"BN batch stats")) return -1;
    if((b==nullptr)!=(xb==nullptr)) return fail_message("BN second view requires matching workspace");
    if(b && (validate_buffer(ctx,b,n,"BN second view") || validate_buffer(ctx,xb,n,"BN second view workspace"))) return -1;
    if(groups>1 && validate_i32_buffer(ctx,ids,rows,"BN buckets")) return -1;
    if(set_context_device(ctx)) return -1;
    bn_forward_kernel<<<static_cast<unsigned>(c),256,0,ctx->stream>>>(a->ptr,b?b->ptr:nullptr,xa->ptr,xb?xb->ptr:nullptr,
        ids?ids->ptr:nullptr,params->ptr,running->ptr,stats->ptr,rows,stride,width,groups,epsilon,momentum,training!=0);
    return check_kernel_launch("BN forward");
}

extern "C" int bulletou_bn_backward(BulletOuCudaCppContext* ctx,
    BulletOuCudaCppF32Buffer* da,BulletOuCudaCppF32Buffer* db,
    BulletOuCudaCppF32Buffer* xa,BulletOuCudaCppF32Buffer* xb,
    BulletOuCudaCppI32Buffer* ids,BulletOuCudaCppF32Buffer* params,
    BulletOuCudaCppF32Buffer* stats,BulletOuCudaCppF32Buffer* grads,
    size_t rows,size_t stride,size_t width,size_t groups) {
    if(!rows || !width || stride<width || !groups || groups>65536/width || rows>SIZE_MAX/stride)
        return fail_message("invalid BN backward shape");
    const size_t n=rows*stride,c=groups*width;
    if(validate_buffer(ctx,da,n,"BN gradient") || validate_buffer(ctx,xa,n,"BN normalized input") ||
       validate_buffer(ctx,params,2*c,"BN affine") || validate_buffer(ctx,stats,2*c,"BN batch stats") ||
       validate_buffer(ctx,grads,2*c,"BN affine gradients")) return -1;
    if((db==nullptr)!=(xb==nullptr)) return fail_message("BN backward second view requires matching workspace");
    if(db && (validate_buffer(ctx,db,n,"BN second gradient") || validate_buffer(ctx,xb,n,"BN second normalized input"))) return -1;
    if(groups>1 && validate_i32_buffer(ctx,ids,rows,"BN buckets")) return -1;
    if(set_context_device(ctx)) return -1;
    bn_backward_kernel<<<static_cast<unsigned>(c),256,0,ctx->stream>>>(da->ptr,db?db->ptr:nullptr,xa->ptr,xb?xb->ptr:nullptr,
        ids?ids->ptr:nullptr,params->ptr,stats->ptr,grads->ptr,rows,stride,width,groups);
    return check_kernel_launch("BN backward");
}

int bn_forward_bound(BulletOuCudaCppContext* ctx,int layer,float* a,float* b,
    const int* ids,size_t rows,size_t stride) {
    const auto c=ctx->bn[layer];if(!c.params)return 0;
    bn_forward_kernel<<<static_cast<unsigned>(c.width*c.groups),256,0,ctx->stream>>>(
        a,b,c.xa,c.xb,ids,c.params,c.running,c.stats,rows,stride,c.width,c.groups,c.epsilon,c.momentum,c.training);
    return check_kernel_launch("SFNN BN forward");
}
int bn_backward_bound(BulletOuCudaCppContext* ctx,int layer,float* a,float* b,
    const int* ids,size_t rows,size_t stride) {
    const auto c=ctx->bn[layer];if(!c.params)return 0;
    bn_backward_kernel<<<static_cast<unsigned>(c.width*c.groups),256,0,ctx->stream>>>(
        a,b,c.xa,c.xb,ids,c.params,c.stats,c.grads,rows,stride,c.width,c.groups);
    return check_kernel_launch("SFNN BN backward");
}
extern "C" int bulletou_bn_bind(BulletOuCudaCppContext* ctx,int layer,
    BulletOuCudaCppF32Buffer* params,BulletOuCudaCppF32Buffer* running,BulletOuCudaCppF32Buffer* stats,
    BulletOuCudaCppF32Buffer* grads,BulletOuCudaCppF32Buffer* xa,BulletOuCudaCppF32Buffer* xb,
    size_t rows,size_t stride,size_t width,size_t groups,float epsilon,float momentum,int training) {
    if(layer<0 || layer>2)return fail_message("invalid BN layer");
    if(!params) {ctx->bn[layer]=BnConfig{};return 0;}
    if(!rows || !width || stride<width || !groups || groups>65536/width || rows>SIZE_MAX/stride ||
        !std::isfinite(epsilon) || epsilon<=0 || !std::isfinite(momentum) || momentum<=0 || momentum>1)
        return fail_message("invalid BN binding");
    size_t c=width*groups,n=rows*stride;
    if(validate_buffer(ctx,params,2*c,"BN params") || validate_buffer(ctx,running,3*c,"BN running") ||
       validate_buffer(ctx,stats,2*c,"BN stats") || validate_buffer(ctx,grads,2*c,"BN gradients") ||
       validate_buffer(ctx,xa,n,"BN workspace") || (layer==0 && validate_buffer(ctx,xb,n,"BN second workspace")))return -1;
    ctx->bn[layer]=BnConfig{params->ptr,running->ptr,stats->ptr,grads->ptr,xa->ptr,xb?xb->ptr:nullptr,width,groups,epsilon,momentum,training!=0};
    return 0;
}
