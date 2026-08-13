#include <stddef.h>
#include <shtns.h>

#include <complex.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#if SHTNS_INTERFACE != 0x307A0
#error "fixtures must be built against the SHTns 3.7 interface (0x307A0)"
#endif

#ifndef GENERATOR_SOURCE_PATH
#define GENERATOR_SOURCE_PATH "reference/shtns37/generate.c"
#endif

#define UPSTREAM_COMMIT "4e04fba84ea156974df5edaf4ee856c0f4f86e77"
#define UPSTREAM_ARCHIVE_SHA256 "5c6a2d585211232a030c6fbbb08f6a794dd1aab987d31511ef53deea12138d97"

typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    unsigned char block[64];
    size_t used;
} sha256_ctx;

static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }

static void sha256_transform(sha256_ctx *ctx, const unsigned char block[64]) {
    static const uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)block[4*i] << 24) | ((uint32_t)block[4*i+1] << 16) |
               ((uint32_t)block[4*i+2] << 8) | (uint32_t)block[4*i+3];
    }
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=ctx->state[0], b=ctx->state[1], c=ctx->state[2], d=ctx->state[3];
    uint32_t e=ctx->state[4], f=ctx->state[5], g=ctx->state[6], h=ctx->state[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t s1=rotr(e,6)^rotr(e,11)^rotr(e,25), ch=(e&f)^((~e)&g);
        uint32_t t1=h+s1+ch+k[i]+w[i], s0=rotr(a,2)^rotr(a,13)^rotr(a,22);
        uint32_t maj=(a&b)^(a&c)^(b&c), t2=s0+maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    ctx->state[0]+=a; ctx->state[1]+=b; ctx->state[2]+=c; ctx->state[3]+=d;
    ctx->state[4]+=e; ctx->state[5]+=f; ctx->state[6]+=g; ctx->state[7]+=h;
}

static void sha256_init(sha256_ctx *ctx) {
    static const uint32_t initial[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    };
    memcpy(ctx->state, initial, sizeof(initial));
    ctx->bit_count = 0; ctx->used = 0;
}

static void sha256_update(sha256_ctx *ctx, const unsigned char *data, size_t len) {
    ctx->bit_count += (uint64_t)len * 8;
    while (len) {
        size_t take = 64 - ctx->used;
        if (take > len) take = len;
        memcpy(ctx->block + ctx->used, data, take);
        ctx->used += take; data += take; len -= take;
        if (ctx->used == 64) { sha256_transform(ctx, ctx->block); ctx->used = 0; }
    }
}

static void sha256_final(sha256_ctx *ctx, unsigned char out[32]) {
    ctx->block[ctx->used++] = 0x80;
    if (ctx->used > 56) {
        while (ctx->used < 64) ctx->block[ctx->used++] = 0;
        sha256_transform(ctx, ctx->block); ctx->used = 0;
    }
    while (ctx->used < 56) ctx->block[ctx->used++] = 0;
    for (int i = 7; i >= 0; --i) ctx->block[ctx->used++] = (unsigned char)(ctx->bit_count >> (8*i));
    sha256_transform(ctx, ctx->block);
    for (int i = 0; i < 8; ++i) {
        out[4*i]=(unsigned char)(ctx->state[i]>>24); out[4*i+1]=(unsigned char)(ctx->state[i]>>16);
        out[4*i+2]=(unsigned char)(ctx->state[i]>>8); out[4*i+3]=(unsigned char)ctx->state[i];
    }
}

static int sha256_file(const char *path, char hex[65]) {
    FILE *file = fopen(path, "rb");
    if (!file) return -1;
    sha256_ctx ctx; sha256_init(&ctx);
    unsigned char buffer[8192]; size_t n;
    while ((n = fread(buffer, 1, sizeof(buffer), file)) != 0) sha256_update(&ctx, buffer, n);
    if (ferror(file)) { fclose(file); return -1; }
    fclose(file);
    unsigned char digest[32]; sha256_final(&ctx, digest);
    static const char digits[] = "0123456789abcdef";
    for (int i = 0; i < 32; ++i) { hex[2*i]=digits[digest[i]>>4]; hex[2*i+1]=digits[digest[i]&15]; }
    hex[64] = '\0'; return 0;
}

static int host_is_little_endian(void) { uint16_t x=1; return *(unsigned char *)&x == 1; }
static uint64_t bswap64(uint64_t x) {
    return ((x & UINT64_C(0x00000000000000ff)) << 56) | ((x & UINT64_C(0x000000000000ff00)) << 40) |
           ((x & UINT64_C(0x0000000000ff0000)) << 24) | ((x & UINT64_C(0x00000000ff000000)) << 8) |
           ((x & UINT64_C(0x000000ff00000000)) >> 8) | ((x & UINT64_C(0x0000ff0000000000)) >> 24) |
           ((x & UINT64_C(0x00ff000000000000)) >> 40) | ((x & UINT64_C(0xff00000000000000)) >> 56);
}
static uint32_t bswap32(uint32_t x) {
    return ((x & UINT32_C(0x000000ff)) << 24) | ((x & UINT32_C(0x0000ff00)) << 8) |
           ((x & UINT32_C(0x00ff0000)) >> 8) | ((x & UINT32_C(0xff000000)) >> 24);
}

static int write_double_le(FILE *file, double value) {
    uint64_t bits; memcpy(&bits, &value, sizeof(bits));
    if (!host_is_little_endian()) bits = bswap64(bits);
    return fwrite(&bits, sizeof(bits), 1, file) == 1 ? 0 : -1;
}
static int write_float_le(FILE *file, float value) {
    uint32_t bits; memcpy(&bits,&value,sizeof(bits)); if(!host_is_little_endian()) bits=bswap32(bits);
    return fwrite(&bits,sizeof(bits),1,file)==1 ? 0 : -1;
}

static int write_spatial_real_file(const char *path, const double *values, int nlat, int nphi) {
    FILE *file = fopen(path, "wb"); if (!file) return -1;
    for (int ip=0; ip<nphi; ++ip) for (int it=0; it<nlat; ++it) {
        if (write_double_le(file, values[(size_t)it*nphi + ip])) { fclose(file); return -1; }
    }
    return fclose(file);
}

static int write_real_file(const char *path, const double *values, size_t count) {
    FILE *file = fopen(path, "wb"); if (!file) return -1;
    for (size_t i=0; i<count; ++i) if (write_double_le(file,values[i])) { fclose(file); return -1; }
    return fclose(file);
}

static int write_spatial_complex_file(const char *path, const cplx *values, int nlat, int nphi) {
    FILE *file = fopen(path, "wb"); if (!file) return -1;
    for (int ip=0; ip<nphi; ++ip) for (int it=0; it<nlat; ++it) {
        cplx value=values[(size_t)it*nphi+ip];
        if (write_double_le(file,creal(value)) || write_double_le(file,cimag(value))) { fclose(file); return -1; }
    }
    return fclose(file);
}

static int write_complex_file(const char *path, const cplx *values, size_t count) {
    FILE *file = fopen(path, "wb"); if (!file) return -1;
    for (size_t i=0; i<count; ++i) {
        if (write_double_le(file, creal(values[i])) || write_double_le(file, cimag(values[i]))) {
            fclose(file); return -1;
        }
    }
    return fclose(file);
}

static int write_complex_float_file(const char *path,const cplx *values,size_t count) {
    FILE *file=fopen(path,"wb"); if(!file)return -1;
    for(size_t i=0;i<count;++i) if(write_float_le(file,(float)creal(values[i])) || write_float_le(file,(float)cimag(values[i]))) { fclose(file); return -1; }
    return fclose(file);
}
static int write_spatial_float_file(const char *path,const double *values,int nlat,int nphi) {
    FILE *file=fopen(path,"wb"); if(!file)return -1;
    for(int ip=0;ip<nphi;++ip) for(int it=0;it<nlat;++it) if(write_float_le(file,(float)values[(size_t)it*nphi+ip])) { fclose(file); return -1; }
    return fclose(file);
}

static void path_join(char *out, size_t cap, const char *dir, const char *name) {
    if (snprintf(out, cap, "%s/%s", dir, name) >= (int)cap) { fputs("output path too long\n", stderr); exit(2); }
}

static int make_directories(const char *path) {
    char copy[4096];
    if (strlen(path) >= sizeof(copy)) { errno = ENAMETOOLONG; return -1; }
    strcpy(copy, path);
    for (char *p = copy + 1; *p; ++p) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(copy, 0777) != 0 && errno != EEXIST) return -1;
        *p = '/';
    }
    return mkdir(copy, 0777) == 0 || errno == EEXIST ? 0 : -1;
}

static void begin_fixture(FILE *manifest, const char *id, const char *capability,
                          const char *grid, const char *norm, int cs_phase,
                          int real_norm, const char *precision, int lmax, int mmax,
                          int mres, int ltr, int nlat, int nphi) {
    const char *tol = strcmp(precision,"float32") == 0 ? "8.0e-5" : "8.0e-12";
    fprintf(manifest,
        "\n[[fixture]]\n"
        "id = \"%s\"\ncapability = \"%s\"\ngrid = \"%s\"\nnorm = \"%s\"\n"
        "cs_phase = %s\nreal_norm = %s\nprecision = \"%s\"\n"
        "lmax = %d\nmmax = %d\nmres = %d\nltr = %d\nnlat = %d\nnphi = %d\n"
        "atol = %s\nrtol = %s\n",
        id,capability,grid,norm,cs_phase?"true":"false",real_norm?"true":"false",
        precision,lmax,mmax,mres,ltr,nlat,nphi,tol,tol);
}

static void payload_block(FILE *manifest, const char *output_dir, const char *name,
                          const char *file, const char *eltype, const int *shape,
                          int ndims, size_t bytes) {
    char path[4096], hash[65]; path_join(path,sizeof(path),output_dir,file);
    if (sha256_file(path,hash)) { perror("payload sha256"); exit(1); }
    fprintf(manifest,"\n[[fixture.payload]]\nname = \"%s\"\nfile = \"%s\"\n"
                     "endian = \"little\"\neltype = \"%s\"\nshape = [",name,file,eltype);
    for (int i=0;i<ndims;++i) fprintf(manifest,"%s%d",i?", ":"",shape[i]);
    fprintf(manifest,"]\nbytes = %zu\nsha256 = \"%s\"\n",bytes,hash);
}

static void fill_real_coefficients(shtns_cfg cfg, cplx *q, double factor) {
    for (unsigned lm=0; lm<cfg->nlm; ++lm) {
        int l=cfg->li[lm], m=cfg->mi[lm];
        double re=factor*(0.021*(l+1)-0.013*(m+1));
        double im=m==0 ? 0.0 : factor*(0.009*(l+1)+0.007*(m+1));
        q[lm]=re+im*I;
    }
}

static void convert_toroidal_to_shtnskit(cplx *t,size_t count) {
    for(size_t i=0;i<count;++i) t[i] = -t[i];
}

static void fill_complex_coefficients(shtns_cfg cfg, cplx *a, double factor) {
    for (int l=0;l<=cfg->lmax;++l) for (int m=-l;m<=l;++m) {
        int lm=LM_cplx(cfg,l,m);
        cplx value=factor*(0.017*(l+1)+0.011*m)+factor*(0.008*(l+1)-0.006*m)*I;
        /* SHTns 3.7's public complex transform applies (-1)^|m| to negative
           orders relative to its documented LM_cplx storage. Store the
           SHTnsKit-compatible coefficient convention in the fixture. */
        a[lm]=(m<0 && ((-m)&1)) ? -value : value;
    }
}

static shtns_cfg make_cfg(int lmax,int mmax,int mres,int norm,int grid,int nlat,int nphi,int layout) {
    if (mmax % mres != 0) { fputs("physical mmax must be divisible by mres\n",stderr); exit(1); }
    shtns_cfg cfg=shtns_create(lmax,mmax/mres,mres,(enum shtns_norm)norm);
    if (!cfg || shtns_set_grid(cfg,(enum shtns_type)(grid|layout),1e-12,nlat,nphi)<0) {
        fputs("SHTns configuration failed\n",stderr); exit(1);
    }
    return cfg;
}

static void generate_scalar_family(FILE *manifest, const char *output_dir) {
    char path[4096], file[256];

    { /* full complex synthesis */
        int lmax=3,mmax=3,mres=1,nlat=8,nphi=10;
        shtns_cfg cfg=make_cfg(lmax,mmax,mres,sht_orthonormal,sht_gauss,nlat,nphi,SHT_PHI_CONTIGUOUS);
        cplx *a=calloc(cfg->nlm_cplx,sizeof(*a)), *z=calloc((size_t)nlat*nphi,sizeof(*z));
        fill_complex_coefficients(cfg,a,1.0); SH_to_spat_cplx(cfg,a,z);
        for(int l=0;l<=lmax;++l) for(int m=-l;m<0;++m) if((-m)&1) a[LM_cplx(cfg,l,m)] = -a[LM_cplx(cfg,l,m)];
        strcpy(file,"scalar_complex_full_coefficients.bin"); path_join(path,sizeof(path),output_dir,file); write_complex_file(path,a,cfg->nlm_cplx);
        strcpy(file,"scalar_complex_full_field.bin"); path_join(path,sizeof(path),output_dir,file); write_spatial_complex_file(path,z,nlat,nphi);
        begin_fixture(manifest,"scalar_complex_full","scalar_complex_full","gauss","orthonormal",1,0,"float64",lmax,mmax,mres,lmax,nlat,nphi);
        int s1[]={ (int)cfg->nlm_cplx }, s2[]={nlat,nphi};
        payload_block(manifest,output_dir,"coefficients","scalar_complex_full_coefficients.bin","complex64",s1,1,(size_t)cfg->nlm_cplx*16);
        payload_block(manifest,output_dir,"field","scalar_complex_full_field.bin","complex64",s2,2,(size_t)nlat*nphi*16);
        free(a); free(z); shtns_destroy(cfg);
    }
    { /* degree-truncated scalar synthesis */
        int lmax=4,mmax=4,mres=1,ltr=2,nlat=8,nphi=10;
        shtns_cfg cfg=make_cfg(lmax,mmax,mres,sht_fourpi|SHT_NO_CS_PHASE,sht_gauss_fly,nlat,nphi,SHT_PHI_CONTIGUOUS);
        cplx *q=calloc(cfg->nlm,sizeof(*q)); double *v=calloc((size_t)nlat*nphi,sizeof(*v));
        fill_real_coefficients(cfg,q,1.0); SH_to_spat_l(cfg,q,v,ltr);
        path_join(path,sizeof(path),output_dir,"scalar_l_coefficients.bin"); write_complex_file(path,q,cfg->nlm);
        path_join(path,sizeof(path),output_dir,"scalar_l_field.bin"); write_spatial_real_file(path,v,nlat,nphi);
        begin_fixture(manifest,"scalar_l","scalar_l","gauss_fly","fourpi",0,0,"float64",lmax,mmax,mres,ltr,nlat,nphi);
        int s1[]={ (int)cfg->nlm },s2[]={nlat,nphi};
        payload_block(manifest,output_dir,"coefficients","scalar_l_coefficients.bin","complex64",s1,1,(size_t)cfg->nlm*16);
        payload_block(manifest,output_dir,"field","scalar_l_field.bin","float64",s2,2,(size_t)nlat*nphi*8);
        free(q); free(v); shtns_destroy(cfg);
    }
    { /* fixed stored order, mres=2 */
        int lmax=4,mmax=4,mres=2,ltr=4,im=1,m=im*mres,nlat=10,nphi=10, count=ltr-m+1;
        shtns_cfg cfg=make_cfg(lmax,mmax,mres,sht_schmidt|SHT_REAL_NORM,sht_reg_dct,nlat,nphi,SHT_PHI_CONTIGUOUS);
        cplx *q=calloc((size_t)lmax+1-m,sizeof(*q)), *v=calloc(nlat,sizeof(*v));
        for(int i=0;i<count;++i) q[i]=(0.03*(i+1)-0.01)+(0.012*(i+1))*I;
        SH_to_spat_ml(cfg,im,q,v,ltr);
        path_join(path,sizeof(path),output_dir,"scalar_ml_coefficients.bin"); write_complex_file(path,q,count);
        path_join(path,sizeof(path),output_dir,"scalar_ml_latitude.bin"); write_complex_file(path,v,nlat);
        begin_fixture(manifest,"scalar_ml","scalar_ml","regular","schmidt",1,1,"float64",lmax,mmax,mres,ltr,nlat,nphi);
        fprintf(manifest,"stored_im = %d\n",im);
        fprintf(manifest,"fixed_mode_scale = %d\n",nphi);
        int s1[]={count},s2[]={nlat};
        payload_block(manifest,output_dir,"coefficients","scalar_ml_coefficients.bin","complex64",s1,1,(size_t)count*16);
        payload_block(manifest,output_dir,"field","scalar_ml_latitude.bin","complex64",s2,1,(size_t)nlat*16);
        free(q); free(v); shtns_destroy(cfg);
    }
    { /* two-field public batch */
        int lmax=3,mmax=3,mres=1,nlat=8,nphi=10,batch=2;
        shtns_cfg cfg=shtns_create(lmax,mmax,mres,sht_orthonormal|SHT_REAL_NORM);
        if(!cfg || shtns_set_many(cfg,batch,0)!=batch || shtns_set_grid(cfg,sht_gauss|SHT_THETA_CONTIGUOUS,1e-12,nlat,nphi)<0) { fputs("batch setup failed\n",stderr); exit(1); }
        cplx *q=calloc((size_t)cfg->nlm*batch,sizeof(*q)); double *v=calloc((size_t)nlat*nphi*batch,sizeof(*v));
        fill_real_coefficients(cfg,q,1.0); fill_real_coefficients(cfg,q+cfg->nlm,-0.7); SH_to_spat(cfg,q,v);
        path_join(path,sizeof(path),output_dir,"scalar_batch_coefficients.bin"); write_complex_file(path,q,(size_t)cfg->nlm*batch);
        path_join(path,sizeof(path),output_dir,"scalar_batch_field.bin"); write_real_file(path,v,(size_t)nlat*nphi*batch);
        begin_fixture(manifest,"scalar_batch","scalar_batch","gauss","orthonormal",1,1,"float64",lmax,mmax,mres,lmax,nlat,nphi);
        int s1[]={ (int)cfg->nlm,batch },s2[]={nlat,nphi,batch};
        payload_block(manifest,output_dir,"coefficients","scalar_batch_coefficients.bin","complex64",s1,2,(size_t)cfg->nlm*batch*16);
        payload_block(manifest,output_dir,"field","scalar_batch_field.bin","float64",s2,3,(size_t)nlat*nphi*batch*8);
        free(q); free(v); shtns_destroy(cfg);
    }
    { /* packed layout is pinned separately from dense wrappers */
        int lmax=3,mmax=3,mres=1,nlat=8,nphi=10;
        shtns_cfg cfg=make_cfg(lmax,mmax,mres,sht_orthonormal,sht_reg_poles,nlat,nphi,SHT_PHI_CONTIGUOUS);
        cplx *q=calloc(cfg->nlm,sizeof(*q)); double *v=calloc((size_t)nlat*nphi,sizeof(*v));
        fill_real_coefficients(cfg,q,0.6); SH_to_spat(cfg,q,v);
        path_join(path,sizeof(path),output_dir,"packed_storage_coefficients.bin"); write_complex_file(path,q,cfg->nlm);
        path_join(path,sizeof(path),output_dir,"packed_storage_field.bin"); write_spatial_real_file(path,v,nlat,nphi);
        begin_fixture(manifest,"packed_storage","packed_storage","regular_poles","orthonormal",1,0,"float64",lmax,mmax,mres,lmax,nlat,nphi);
        int s1[]={ (int)cfg->nlm },s2[]={nlat,nphi};
        payload_block(manifest,output_dir,"coefficients","packed_storage_coefficients.bin","complex64",s1,1,(size_t)cfg->nlm*16);
        payload_block(manifest,output_dir,"field","packed_storage_field.bin","float64",s2,2,(size_t)nlat*nphi*8);
        free(q); free(v); shtns_destroy(cfg);
    }
}

static void write_packed_payload(FILE *manifest,const char *out,const char *name,
                                 const char *file,const cplx *values,int count,
                                 int fp32) {
    char path[4096]; path_join(path,sizeof(path),out,file);
    if(fp32) write_complex_float_file(path,values,count); else write_complex_file(path,values,count);
    int shape[]={count}; payload_block(manifest,out,name,file,fp32?"complex32":"complex64",shape,1,(size_t)count*(fp32?8:16));
}
static void write_spatial_payload(FILE *manifest,const char *out,const char *name,
                                  const char *file,const double *values,int nlat,
                                  int nphi,int fp32) {
    char path[4096]; path_join(path,sizeof(path),out,file);
    if(fp32) write_spatial_float_file(path,values,nlat,nphi); else write_spatial_real_file(path,values,nlat,nphi);
    int shape[]={nlat,nphi}; payload_block(manifest,out,name,file,fp32?"float32":"float64",shape,2,(size_t)nlat*nphi*(fp32?4:8));
}
static void write_mode_payload(FILE *manifest,const char *out,const char *name,
                               const char *file,const cplx *values,int count) {
    char path[4096]; path_join(path,sizeof(path),out,file); write_complex_file(path,values,count);
    int shape[]={count}; payload_block(manifest,out,name,file,"complex64",shape,1,(size_t)count*16);
}

static void generate_vector_family(FILE *manifest,const char *out) {
    { /* full, Float32 is a documented downcast of the FP64 CPU oracle */
        int lmax=3,mmax=3,mres=1,nlat=8,nphi=10,fp32=1;
        shtns_cfg cfg=make_cfg(lmax,mmax,mres,sht_orthonormal,sht_gauss,nlat,nphi,SHT_PHI_CONTIGUOUS);
        cplx *s=calloc(cfg->nlm,sizeof(*s)),*t=calloc(cfg->nlm,sizeof(*t));
        double *vt=calloc((size_t)nlat*nphi,sizeof(*vt)),*vp=calloc((size_t)nlat*nphi,sizeof(*vp));
        fill_real_coefficients(cfg,s,0.7); fill_real_coefficients(cfg,t,-0.45); s[0]=t[0]=0; SHsphtor_to_spat(cfg,s,t,vt,vp); convert_toroidal_to_shtnskit(t,cfg->nlm);
        begin_fixture(manifest,"sphtor_full_f32","sphtor_full","gauss","orthonormal",1,0,"float32",lmax,mmax,mres,lmax,nlat,nphi);
        fprintf(manifest,"precision_provenance = \"little-endian Float32 downcast of independent SHTns 3.7 FP64 oracle\"\n");
        write_packed_payload(manifest,out,"S","sphtor_full_f32_S.bin",s,cfg->nlm,fp32);
        write_packed_payload(manifest,out,"T","sphtor_full_f32_T.bin",t,cfg->nlm,fp32);
        write_spatial_payload(manifest,out,"Vt","sphtor_full_f32_Vt.bin",vt,nlat,nphi,fp32);
        write_spatial_payload(manifest,out,"Vp","sphtor_full_f32_Vp.bin",vp,nlat,nphi,fp32);
        free(s);free(t);free(vt);free(vp);shtns_destroy(cfg);
    }
    { /* l-truncated */
        int lmax=4,mmax=4,mres=1,ltr=2,nlat=10,nphi=12;
        shtns_cfg cfg=make_cfg(lmax,mmax,mres,sht_fourpi|SHT_NO_CS_PHASE,sht_reg_dct,nlat,nphi,SHT_PHI_CONTIGUOUS);
        cplx *s=calloc(cfg->nlm,sizeof(*s)),*t=calloc(cfg->nlm,sizeof(*t));double *vt=calloc((size_t)nlat*nphi,sizeof(*vt)),*vp=calloc((size_t)nlat*nphi,sizeof(*vp));
        fill_real_coefficients(cfg,s,0.5);fill_real_coefficients(cfg,t,-0.3);s[0]=t[0]=0;SHsphtor_to_spat_l(cfg,s,t,vt,vp,ltr);convert_toroidal_to_shtnskit(t,cfg->nlm);
        begin_fixture(manifest,"sphtor_l","sphtor_l","regular","fourpi",0,0,"float64",lmax,mmax,mres,ltr,nlat,nphi);
        write_packed_payload(manifest,out,"S","sphtor_l_S.bin",s,cfg->nlm,0);write_packed_payload(manifest,out,"T","sphtor_l_T.bin",t,cfg->nlm,0);
        write_spatial_payload(manifest,out,"Vt","sphtor_l_Vt.bin",vt,nlat,nphi,0);write_spatial_payload(manifest,out,"Vp","sphtor_l_Vp.bin",vp,nlat,nphi,0);
        free(s);free(t);free(vt);free(vp);shtns_destroy(cfg);
    }
    { /* fixed stored order */
        int lmax=4,mmax=4,mres=2,ltr=4,im=1,m=2,nlat=10,nphi=10,count=ltr-m+1;
        shtns_cfg cfg=make_cfg(lmax,mmax,mres,sht_schmidt|SHT_REAL_NORM,sht_reg_poles,nlat,nphi,SHT_PHI_CONTIGUOUS);
        cplx *s=calloc(count,sizeof(*s)),*t=calloc(count,sizeof(*t)),*vt=calloc(nlat,sizeof(*vt)),*vp=calloc(nlat,sizeof(*vp));
        for(int i=0;i<count;++i){s[i]=0.02*(i+1)+0.01*(i+2)*I;t[i]=-0.015*(i+1)+0.007*(i+1)*I;} SHsphtor_to_spat_ml(cfg,im,s,t,vt,vp,ltr);convert_toroidal_to_shtnskit(t,count);
        begin_fixture(manifest,"sphtor_ml","sphtor_ml","regular_poles","schmidt",1,1,"float64",lmax,mmax,mres,ltr,nlat,nphi);
        fprintf(manifest,"stored_im = %d\nfixed_mode_scale = %d\n",im,nphi);
        write_mode_payload(manifest,out,"S","sphtor_ml_S.bin",s,count);write_mode_payload(manifest,out,"T","sphtor_ml_T.bin",t,count);
        write_mode_payload(manifest,out,"Vt","sphtor_ml_Vt.bin",vt,nlat);write_mode_payload(manifest,out,"Vp","sphtor_ml_Vp.bin",vp,nlat);
        free(s);free(t);free(vt);free(vp);shtns_destroy(cfg);
    }
    { /* batch fixture is two independent public SHTns calls */
        int lmax=3,mmax=3,mres=1,nlat=8,nphi=10,batch=2,nlm=(lmax+1)*(lmax+2)/2;
        shtns_cfg cfg=make_cfg(lmax,mmax,mres,sht_orthonormal|SHT_REAL_NORM,sht_gauss_fly,nlat,nphi,SHT_PHI_CONTIGUOUS);
        cplx *s=calloc((size_t)nlm*batch,sizeof(*s)),*t=calloc((size_t)nlm*batch,sizeof(*t));
        double *vt=calloc((size_t)nlat*nphi*batch,sizeof(*vt)),*vp=calloc((size_t)nlat*nphi*batch,sizeof(*vp));
        double *tmp1=calloc((size_t)nlat*nphi,sizeof(*tmp1)),*tmp2=calloc((size_t)nlat*nphi,sizeof(*tmp2));
        for(int k=0;k<batch;++k){fill_real_coefficients(cfg,s+(size_t)k*nlm,0.4+0.2*k);fill_real_coefficients(cfg,t+(size_t)k*nlm,-0.25-0.1*k);s[(size_t)k*nlm]=t[(size_t)k*nlm]=0;SHsphtor_to_spat(cfg,s+(size_t)k*nlm,t+(size_t)k*nlm,tmp1,tmp2);convert_toroidal_to_shtnskit(t+(size_t)k*nlm,nlm);for(int ip=0;ip<nphi;++ip)for(int it=0;it<nlat;++it){size_t dst=it+(size_t)nlat*ip+(size_t)nlat*nphi*k,src=(size_t)it*nphi+ip;vt[dst]=tmp1[src];vp[dst]=tmp2[src];}}
        begin_fixture(manifest,"sphtor_batch","sphtor_batch","gauss_fly","orthonormal",1,1,"float64",lmax,mmax,mres,lmax,nlat,nphi);
        char path[4096];path_join(path,sizeof(path),out,"sphtor_batch_S.bin");write_complex_file(path,s,(size_t)nlm*batch);path_join(path,sizeof(path),out,"sphtor_batch_T.bin");write_complex_file(path,t,(size_t)nlm*batch);path_join(path,sizeof(path),out,"sphtor_batch_Vt.bin");write_real_file(path,vt,(size_t)nlat*nphi*batch);path_join(path,sizeof(path),out,"sphtor_batch_Vp.bin");write_real_file(path,vp,(size_t)nlat*nphi*batch);
        int ss[]={nlm,batch},sv[]={nlat,nphi,batch};payload_block(manifest,out,"S","sphtor_batch_S.bin","complex64",ss,2,(size_t)nlm*batch*16);payload_block(manifest,out,"T","sphtor_batch_T.bin","complex64",ss,2,(size_t)nlm*batch*16);payload_block(manifest,out,"Vt","sphtor_batch_Vt.bin","float64",sv,3,(size_t)nlat*nphi*batch*8);payload_block(manifest,out,"Vp","sphtor_batch_Vp.bin","float64",sv,3,(size_t)nlat*nphi*batch*8);
        free(s);free(t);free(vt);free(vp);free(tmp1);free(tmp2);shtns_destroy(cfg);
    }
}

static void generate_qst_family(FILE *manifest,const char *out) {
    { int lmax=3,mmax=3,mres=1,nlat=8,nphi=10;
      shtns_cfg cfg=make_cfg(lmax,mmax,mres,sht_orthonormal,sht_gauss,nlat,nphi,SHT_PHI_CONTIGUOUS);
      cplx *q=calloc(cfg->nlm,sizeof(*q)),*s=calloc(cfg->nlm,sizeof(*s)),*t=calloc(cfg->nlm,sizeof(*t));
      double *vr=calloc((size_t)nlat*nphi,sizeof(*vr)),*vt=calloc((size_t)nlat*nphi,sizeof(*vt)),*vp=calloc((size_t)nlat*nphi,sizeof(*vp));
      fill_real_coefficients(cfg,q,0.9);fill_real_coefficients(cfg,s,0.55);fill_real_coefficients(cfg,t,-0.35);s[0]=t[0]=0;SHqst_to_spat(cfg,q,s,t,vr,vt,vp);convert_toroidal_to_shtnskit(t,cfg->nlm);
      begin_fixture(manifest,"qst_full","qst_full","gauss","orthonormal",1,0,"float64",lmax,mmax,mres,lmax,nlat,nphi);
      write_packed_payload(manifest,out,"Q","qst_full_Q.bin",q,cfg->nlm,0);write_packed_payload(manifest,out,"S","qst_full_S.bin",s,cfg->nlm,0);write_packed_payload(manifest,out,"T","qst_full_T.bin",t,cfg->nlm,0);write_spatial_payload(manifest,out,"Vr","qst_full_Vr.bin",vr,nlat,nphi,0);write_spatial_payload(manifest,out,"Vt","qst_full_Vt.bin",vt,nlat,nphi,0);write_spatial_payload(manifest,out,"Vp","qst_full_Vp.bin",vp,nlat,nphi,0);
      free(q);free(s);free(t);free(vr);free(vt);free(vp);shtns_destroy(cfg); }
    { int lmax=4,mmax=4,mres=1,ltr=2,nlat=10,nphi=12;
      shtns_cfg cfg=make_cfg(lmax,mmax,mres,sht_fourpi|SHT_NO_CS_PHASE,sht_gauss_fly,nlat,nphi,SHT_PHI_CONTIGUOUS);
      cplx *q=calloc(cfg->nlm,sizeof(*q)),*s=calloc(cfg->nlm,sizeof(*s)),*t=calloc(cfg->nlm,sizeof(*t));double *vr=calloc((size_t)nlat*nphi,sizeof(*vr)),*vt=calloc((size_t)nlat*nphi,sizeof(*vt)),*vp=calloc((size_t)nlat*nphi,sizeof(*vp));
      fill_real_coefficients(cfg,q,0.8);fill_real_coefficients(cfg,s,0.45);fill_real_coefficients(cfg,t,-0.25);s[0]=t[0]=0;SHqst_to_spat_l(cfg,q,s,t,vr,vt,vp,ltr);convert_toroidal_to_shtnskit(t,cfg->nlm);
      begin_fixture(manifest,"qst_l","qst_l","gauss_fly","fourpi",0,0,"float64",lmax,mmax,mres,ltr,nlat,nphi);
      write_packed_payload(manifest,out,"Q","qst_l_Q.bin",q,cfg->nlm,0);write_packed_payload(manifest,out,"S","qst_l_S.bin",s,cfg->nlm,0);write_packed_payload(manifest,out,"T","qst_l_T.bin",t,cfg->nlm,0);write_spatial_payload(manifest,out,"Vr","qst_l_Vr.bin",vr,nlat,nphi,0);write_spatial_payload(manifest,out,"Vt","qst_l_Vt.bin",vt,nlat,nphi,0);write_spatial_payload(manifest,out,"Vp","qst_l_Vp.bin",vp,nlat,nphi,0);
      free(q);free(s);free(t);free(vr);free(vt);free(vp);shtns_destroy(cfg); }
    { int lmax=4,mmax=4,mres=2,ltr=4,im=1,m=2,nlat=10,nphi=10,count=ltr-m+1;
      shtns_cfg cfg=make_cfg(lmax,mmax,mres,sht_schmidt|SHT_REAL_NORM,sht_reg_dct,nlat,nphi,SHT_PHI_CONTIGUOUS);
      cplx *q=calloc(count,sizeof(*q)),*s=calloc(count,sizeof(*s)),*t=calloc(count,sizeof(*t)),*vr=calloc(nlat,sizeof(*vr)),*vt=calloc(nlat,sizeof(*vt)),*vp=calloc(nlat,sizeof(*vp));
      for(int i=0;i<count;++i){q[i]=0.025*(i+1)+0.009*(i+1)*I;s[i]=0.018*(i+1)+0.011*(i+2)*I;t[i]=-0.013*(i+1)+0.006*(i+1)*I;}SHqst_to_spat_ml(cfg,im,q,s,t,vr,vt,vp,ltr);convert_toroidal_to_shtnskit(t,count);
      begin_fixture(manifest,"qst_ml","qst_ml","regular","schmidt",1,1,"float64",lmax,mmax,mres,ltr,nlat,nphi);fprintf(manifest,"stored_im = %d\nfixed_mode_scale = %d\n",im,nphi);
      write_mode_payload(manifest,out,"Q","qst_ml_Q.bin",q,count);write_mode_payload(manifest,out,"S","qst_ml_S.bin",s,count);write_mode_payload(manifest,out,"T","qst_ml_T.bin",t,count);write_mode_payload(manifest,out,"Vr","qst_ml_Vr.bin",vr,nlat);write_mode_payload(manifest,out,"Vt","qst_ml_Vt.bin",vt,nlat);write_mode_payload(manifest,out,"Vp","qst_ml_Vp.bin",vp,nlat);
      free(q);free(s);free(t);free(vr);free(vt);free(vp);shtns_destroy(cfg); }
    { int lmax=3,mmax=3,mres=1,nlat=8,nphi=10,batch=2,nlm=10;
      shtns_cfg cfg=make_cfg(lmax,mmax,mres,sht_orthonormal|SHT_REAL_NORM,sht_reg_poles,nlat,nphi,SHT_PHI_CONTIGUOUS);
      cplx *q=calloc((size_t)nlm*batch,sizeof(*q)),*s=calloc((size_t)nlm*batch,sizeof(*s)),*t=calloc((size_t)nlm*batch,sizeof(*t));double *vr=calloc((size_t)nlat*nphi*batch,sizeof(*vr)),*vt=calloc((size_t)nlat*nphi*batch,sizeof(*vt)),*vp=calloc((size_t)nlat*nphi*batch,sizeof(*vp)),*a=calloc((size_t)nlat*nphi,sizeof(*a)),*b=calloc((size_t)nlat*nphi,sizeof(*b)),*c=calloc((size_t)nlat*nphi,sizeof(*c));
      for(int k=0;k<batch;++k){fill_real_coefficients(cfg,q+(size_t)k*nlm,0.7+0.1*k);fill_real_coefficients(cfg,s+(size_t)k*nlm,0.35+0.1*k);fill_real_coefficients(cfg,t+(size_t)k*nlm,-0.2-0.05*k);s[(size_t)k*nlm]=t[(size_t)k*nlm]=0;SHqst_to_spat(cfg,q+(size_t)k*nlm,s+(size_t)k*nlm,t+(size_t)k*nlm,a,b,c);convert_toroidal_to_shtnskit(t+(size_t)k*nlm,nlm);for(int ip=0;ip<nphi;++ip)for(int it=0;it<nlat;++it){size_t dst=it+(size_t)nlat*ip+(size_t)nlat*nphi*k,src=(size_t)it*nphi+ip;vr[dst]=a[src];vt[dst]=b[src];vp[dst]=c[src];}}
      begin_fixture(manifest,"qst_batch","qst_batch","regular_poles","orthonormal",1,1,"float64",lmax,mmax,mres,lmax,nlat,nphi);
      char path[4096];const char *names[]={"Q","S","T"};cplx *cs[]={q,s,t};const char *cfiles[]={"qst_batch_Q.bin","qst_batch_S.bin","qst_batch_T.bin"};int ss[]={nlm,batch};for(int i=0;i<3;++i){path_join(path,sizeof(path),out,cfiles[i]);write_complex_file(path,cs[i],(size_t)nlm*batch);payload_block(manifest,out,names[i],cfiles[i],"complex64",ss,2,(size_t)nlm*batch*16);}const char *vnames[]={"Vr","Vt","Vp"};double *vs[]={vr,vt,vp};const char *vfiles[]={"qst_batch_Vr.bin","qst_batch_Vt.bin","qst_batch_Vp.bin"};int sv[]={nlat,nphi,batch};for(int i=0;i<3;++i){path_join(path,sizeof(path),out,vfiles[i]);write_real_file(path,vs[i],(size_t)nlat*nphi*batch);payload_block(manifest,out,vnames[i],vfiles[i],"float64",sv,3,(size_t)nlat*nphi*batch*8);}
      free(q);free(s);free(t);free(vr);free(vt);free(vp);free(a);free(b);free(c);shtns_destroy(cfg); }
}

static void write_values_payload(FILE *manifest,const char *out,const char *name,
                                 const char *file,const double *v,int n) {
    char path[4096];path_join(path,sizeof(path),out,file);write_real_file(path,v,n);int shape[]={n};payload_block(manifest,out,name,file,"float64",shape,1,(size_t)n*8);
}
static void write_complex_values_payload(FILE *manifest,const char *out,const char *name,
                                         const char *file,const cplx *v,int n) {
    char path[4096];path_join(path,sizeof(path),out,file);write_complex_file(path,v,n);int shape[]={n};payload_block(manifest,out,name,file,"complex64",shape,1,(size_t)n*16);
}
static void generate_local_family(FILE *manifest,const char *out) {
    int lmax=3,mmax=3,mres=1,nlat=8,nphi=10;double cost=0.23,phi=0.47;
    shtns_cfg cfg=make_cfg(lmax,mmax,mres,sht_orthonormal,sht_gauss,nlat,nphi,SHT_PHI_CONTIGUOUS);
    cplx *q=calloc(cfg->nlm,sizeof(*q)),*s=calloc(cfg->nlm,sizeof(*s)),*t=calloc(cfg->nlm,sizeof(*t));fill_real_coefficients(cfg,q,0.8);fill_real_coefficients(cfg,s,0.5);fill_real_coefficients(cfg,t,-0.3);s[0]=t[0]=0;
    { double value=SH_to_point(cfg,q,cost,phi);begin_fixture(manifest,"point","point","gauss","orthonormal",1,0,"float64",lmax,mmax,mres,lmax,nlat,nphi);fprintf(manifest,"cost = %.17g\nphi = %.17g\n",cost,phi);write_packed_payload(manifest,out,"Q","point_Q.bin",q,cfg->nlm,0);write_values_payload(manifest,out,"value","point_value.bin",&value,1); }
    cplx *z=calloc(cfg->nlm_cplx,sizeof(*z));fill_complex_coefficients(cfg,z,0.7);
    { cplx value;SH_to_point_cplx(cfg,z,cost,phi,&value);for(int l=0;l<=lmax;++l)for(int m=-l;m<0;++m)if((-m)&1)z[LM_cplx(cfg,l,m)]=-z[LM_cplx(cfg,l,m)];begin_fixture(manifest,"point_complex","point_complex","gauss","orthonormal",1,0,"float64",lmax,mmax,mres,lmax,nlat,nphi);fprintf(manifest,"cost = %.17g\nphi = %.17g\n",cost,phi);write_packed_payload(manifest,out,"A","point_complex_A.bin",z,cfg->nlm_cplx,0);write_complex_values_payload(manifest,out,"value","point_complex_value.bin",&value,1); }
    { double *v=calloc(nphi,sizeof(*v));SH_to_lat(cfg,q,cost,v,nphi,lmax,mmax);begin_fixture(manifest,"latitude","latitude","gauss","orthonormal",1,0,"float64",lmax,mmax,mres,lmax,nlat,nphi);fprintf(manifest,"cost = %.17g\n",cost);write_packed_payload(manifest,out,"Q","latitude_Q.bin",q,cfg->nlm,0);write_values_payload(manifest,out,"values","latitude_values.bin",v,nphi);free(v); }
    { cplx *values=calloc(nphi,sizeof(*values));cplx *zi=calloc(cfg->nlm_cplx,sizeof(*zi));fill_complex_coefficients(cfg,zi,0.7);for(int j=0;j<nphi;++j)SH_to_point_cplx(cfg,zi,cost,2.0*3.14159265358979323846*j/nphi,&values[j]);for(int l=0;l<=lmax;++l)for(int m=-l;m<0;++m)if((-m)&1)zi[LM_cplx(cfg,l,m)]=-zi[LM_cplx(cfg,l,m)];begin_fixture(manifest,"latitude_complex","latitude_complex","gauss","orthonormal",1,0,"float64",lmax,mmax,mres,lmax,nlat,nphi);fprintf(manifest,"cost = %.17g\n",cost);write_packed_payload(manifest,out,"A","latitude_complex_A.bin",zi,cfg->nlm_cplx,0);write_complex_values_payload(manifest,out,"values","latitude_complex_values.bin",values,nphi);free(values);free(zi); }
    { double v[3];SHqst_to_point(cfg,q,s,t,cost,phi,&v[0],&v[1],&v[2]);convert_toroidal_to_shtnskit(t,cfg->nlm);begin_fixture(manifest,"qst_point","qst_point","gauss","orthonormal",1,0,"float64",lmax,mmax,mres,lmax,nlat,nphi);fprintf(manifest,"cost = %.17g\nphi = %.17g\n",cost,phi);write_packed_payload(manifest,out,"Q","qst_point_Q.bin",q,cfg->nlm,0);write_packed_payload(manifest,out,"S","qst_point_S.bin",s,cfg->nlm,0);write_packed_payload(manifest,out,"T","qst_point_T.bin",t,cfg->nlm,0);write_values_payload(manifest,out,"value","qst_point_value.bin",v,3); }
    { double *vr=calloc(nphi,sizeof(*vr)),*vt=calloc(nphi,sizeof(*vt)),*vp=calloc(nphi,sizeof(*vp));convert_toroidal_to_shtnskit(t,cfg->nlm);SHqst_to_lat(cfg,q,s,t,cost,vr,vt,vp,nphi,lmax,mmax);convert_toroidal_to_shtnskit(t,cfg->nlm);begin_fixture(manifest,"qst_latitude","qst_latitude","gauss","orthonormal",1,0,"float64",lmax,mmax,mres,lmax,nlat,nphi);fprintf(manifest,"cost = %.17g\n",cost);write_packed_payload(manifest,out,"Q","qst_latitude_Q.bin",q,cfg->nlm,0);write_packed_payload(manifest,out,"S","qst_latitude_S.bin",s,cfg->nlm,0);write_packed_payload(manifest,out,"T","qst_latitude_T.bin",t,cfg->nlm,0);write_values_payload(manifest,out,"Vr","qst_latitude_Vr.bin",vr,nphi);write_values_payload(manifest,out,"Vt","qst_latitude_Vt.bin",vt,nphi);write_values_payload(manifest,out,"Vp","qst_latitude_Vp.bin",vp,nphi);free(vr);free(vt);free(vp); }
    { double v[3];SH_to_grad_point(cfg,q,s,cost,phi,&v[0],&v[1],&v[2]);begin_fixture(manifest,"gradient_point","gradient_point","gauss","orthonormal",1,0,"float64",lmax,mmax,mres,lmax,nlat,nphi);fprintf(manifest,"cost = %.17g\nphi = %.17g\n",cost,phi);write_packed_payload(manifest,out,"Dr","gradient_point_Dr.bin",q,cfg->nlm,0);write_packed_payload(manifest,out,"S","gradient_point_S.bin",s,cfg->nlm,0);write_values_payload(manifest,out,"value","gradient_point_value.bin",v,3); }
    free(q);free(s);free(t);free(z);shtns_destroy(cfg);
}

static void generate_operator_rotation_family(FILE *manifest,const char *out) {
    int lmax=4,mmax=4,mres=1,nlat=10,nphi=12;
    shtns_cfg cfg=make_cfg(lmax,mmax,mres,sht_orthonormal,sht_gauss,nlat,nphi,SHT_PHI_CONTIGUOUS);
    cplx *q=calloc(cfg->nlm,sizeof(*q));fill_real_coefficients(cfg,q,0.75);
    { double *ct=calloc((size_t)2*cfg->nlm,sizeof(*ct)),*dt=calloc((size_t)2*cfg->nlm,sizeof(*dt)),*dt_source=calloc((size_t)2*cfg->nlm,sizeof(*dt_source));cplx *rct=calloc(cfg->nlm,sizeof(*rct)),*rdt=calloc(cfg->nlm,sizeof(*rdt));mul_ct_matrix(cfg,ct);st_dt_matrix(cfg,dt);SH_mul_mx(cfg,ct,q,rct);SH_mul_mx(cfg,dt,q,rdt);for(unsigned lm=0;lm<cfg->nlm;++lm){int l=cfg->li[lm],m=cfg->mi[lm];dt_source[2*lm]=(l>m)?dt[2*LM(cfg,l-1,m)+1]:0.0;dt_source[2*lm+1]=(l<lmax)?dt[2*LM(cfg,l+1,m)]:0.0;}
      begin_fixture(manifest,"operators","operators","gauss","orthonormal",1,0,"float64",lmax,mmax,mres,lmax,nlat,nphi);write_packed_payload(manifest,out,"Q","operators_Q.bin",q,cfg->nlm,0);write_values_payload(manifest,out,"ct_matrix","operators_ct_matrix.bin",ct,2*cfg->nlm);write_values_payload(manifest,out,"dt_matrix","operators_dt_matrix.bin",dt_source,2*cfg->nlm);write_packed_payload(manifest,out,"ct_result","operators_ct_result.bin",rct,cfg->nlm,0);write_packed_payload(manifest,out,"dt_result","operators_dt_result.bin",rdt,cfg->nlm,0);free(ct);free(dt);free(dt_source);free(rct);free(rdt); }
    { double alpha=0.37,beta=0.41;cplx *z=calloc(cfg->nlm,sizeof(*z)),*y=calloc(cfg->nlm,sizeof(*y));SH_Zrotate(cfg,q,-alpha,z);SH_Yrotate(cfg,q,beta,y);
      begin_fixture(manifest,"rotations","rotations","gauss","orthonormal",1,0,"float64",lmax,mmax,mres,lmax,nlat,nphi);fprintf(manifest,"z_angle = %.17g\ny_angle = %.17g\n",alpha,beta);write_packed_payload(manifest,out,"Q","rotations_Q.bin",q,cfg->nlm,0);write_packed_payload(manifest,out,"Z","rotations_Z.bin",z,cfg->nlm,0);write_packed_payload(manifest,out,"Y","rotations_Y.bin",y,cfg->nlm,0);free(z);free(y); }
    free(q);shtns_destroy(cfg);
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s OUTPUT_DIRECTORY\n", argv[0]); return 2; }
    if (make_directories(argv[1]) != 0) { perror("mkdir"); return 1; }

    const int lmax=2, mmax=2, mres=1, nlat=6, nphi=8;
    shtns_cfg cfg = shtns_create(lmax, mmax, mres, sht_orthonormal);
    if (!cfg) { fputs("shtns_create failed\n", stderr); return 1; }
    if (shtns_set_grid(cfg, sht_gauss | SHT_PHI_CONTIGUOUS, 1e-12, nlat, nphi) < 0) {
        fputs("shtns_set_grid failed\n", stderr); shtns_destroy(cfg); return 1;
    }

    cplx *coefficients = calloc(cfg->nlm, sizeof(*coefficients));
    double *field = calloc(cfg->nspat, sizeof(*field));
    if (!coefficients || !field) { fputs("allocation failed\n", stderr); return 1; }
    coefficients[LM(cfg,0,0)] = 0.31;
    coefficients[LM(cfg,1,0)] = -0.17;
    coefficients[LM(cfg,1,1)] = 0.13 - 0.09*I;
    coefficients[LM(cfg,2,0)] = 0.07;
    coefficients[LM(cfg,2,1)] = -0.08 + 0.04*I;
    coefficients[LM(cfg,2,2)] = 0.05 + 0.03*I;
    SH_to_spat(cfg, coefficients, field);

    char coeff_path[4096], field_path[4096], manifest_path[4096];
    path_join(coeff_path,sizeof(coeff_path),argv[1],"scalar_real_full_gauss_f64_coefficients.bin");
    path_join(field_path,sizeof(field_path),argv[1],"scalar_real_full_gauss_f64_field.bin");
    path_join(manifest_path,sizeof(manifest_path),argv[1],"manifest.toml");
    if (write_complex_file(coeff_path,coefficients,cfg->nlm) ||
        write_spatial_real_file(field_path,field,nlat,nphi)) {
        perror("write payload"); return 1;
    }

    char source_sha[65], coeff_sha[65], field_sha[65];
    if (sha256_file(GENERATOR_SOURCE_PATH,source_sha) || sha256_file(coeff_path,coeff_sha) ||
        sha256_file(field_path,field_sha)) { perror("sha256"); return 1; }
    FILE *manifest=fopen(manifest_path,"wb"); if (!manifest) { perror("manifest"); return 1; }
    fprintf(manifest,
        "format_version = 1\n"
        "shtns_version = \"3.7\"\n"
        "shtns_interface = 0x307A0\n"
        "upstream_tag = \"v3.7\"\n"
        "upstream_commit = \"%s\"\n"
        "upstream_archive_sha256 = \"%s\"\n"
        "generator_source_sha256 = \"%s\"\n\n"
        "[[fixture]]\n"
        "id = \"scalar_real_full_gauss_f64\"\n"
        "capability = \"scalar_real_full\"\n"
        "grid = \"gauss\"\n"
        "norm = \"orthonormal\"\n"
        "cs_phase = true\n"
        "real_norm = false\n"
        "precision = \"float64\"\n"
        "lmax = %d\n"
        "mmax = %d\n"
        "mres = %d\n"
        "ltr = %d\n"
        "nlat = %d\n"
        "nphi = %d\n"
        "atol = 3.0e-12\n"
        "rtol = 3.0e-12\n\n"
        "[[fixture.payload]]\n"
        "name = \"coefficients\"\n"
        "file = \"scalar_real_full_gauss_f64_coefficients.bin\"\n"
        "endian = \"little\"\n"
        "eltype = \"complex64\"\n"
        "shape = [%u]\n"
        "bytes = %zu\n"
        "sha256 = \"%s\"\n\n"
        "[[fixture.payload]]\n"
        "name = \"field\"\n"
        "file = \"scalar_real_full_gauss_f64_field.bin\"\n"
        "endian = \"little\"\n"
        "eltype = \"float64\"\n"
        "shape = [%d, %d]\n"
        "bytes = %zu\n"
        "sha256 = \"%s\"\n",
        UPSTREAM_COMMIT,UPSTREAM_ARCHIVE_SHA256,source_sha,lmax,mmax,mres,lmax,nlat,nphi,
        cfg->nlm,(size_t)cfg->nlm*16,coeff_sha,nlat,nphi,(size_t)nlat*nphi*8,field_sha);
    generate_scalar_family(manifest,argv[1]);
    generate_vector_family(manifest,argv[1]);
    generate_qst_family(manifest,argv[1]);
    generate_local_family(manifest,argv[1]);
    generate_operator_rotation_family(manifest,argv[1]);
    if (fclose(manifest)) { perror("manifest close"); return 1; }

    free(coefficients); free(field); shtns_destroy(cfg);
    printf("wrote 23 SHTns 3.7 fixtures to %s\n",argv[1]);
    return 0;
}
