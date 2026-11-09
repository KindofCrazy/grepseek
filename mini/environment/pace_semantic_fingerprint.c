/* Order-independent SHA256 fingerprints of every term -> canonical posting.
 * Term IDs can differ because upstream discovers per-line terms through a set.
 * Uses the system OpenSSL runtime through dlopen; no new library installation.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct { unsigned char *data; size_t size; } File;
static void fail(const char *message) { fprintf(stderr,"%s\n",message); exit(1); }
static File mapped(const char *directory,const char *name) {
    char path[4096]; snprintf(path,sizeof path,"%s/%s",directory,name);
    int descriptor=open(path,O_RDONLY); struct stat info;
    if(descriptor<0 || fstat(descriptor,&info)) fail("Cannot map index file");
    File f={NULL,(size_t)info.st_size};
    if(f.size) { f.data=mmap(NULL,f.size,PROT_READ,MAP_PRIVATE,descriptor,0);
        if(f.data==MAP_FAILED) fail("Index mmap failed"); madvise(f.data,f.size,MADV_SEQUENTIAL); }
    close(descriptor); return f;
}
static uint32_t u32(const void *p) { uint32_t v; memcpy(&v,p,4); return v; }
static uint64_t u64(const void *p) { uint64_t v; memcpy(&v,p,8); return v; }
static size_t encoded_size(uint64_t count,uint64_t universe) {
    unsigned low=0; while((UINT64_C(1)<<(low+1))<=universe/count) low++;
    return 1+(count*low+7)/8+(count+(universe>>low)+8)/8;
}
static void hexadecimal(const unsigned char *bytes) { for(int i=0;i<32;i++) printf("%02x",bytes[i]); }
int main(int argc,char **argv) {
    if(argc!=5) fail("usage: fingerprint INDEX NUM_TERMS NUM_SINGLETONS NUM_LINES");
    uint64_t terms=strtoull(argv[2],NULL,10),singles=strtoull(argv[3],NULL,10),lines=strtoull(argv[4],NULL,10);
    if(!terms || singles>terms || !lines) fail("Invalid index counts");
    File vocabulary=mapped(argv[1],"vocab.bin"),single=mapped(argv[1],"singles.bin"),
         counts=mapped(argv[1],"multi_count.bin"),offsets=mapped(argv[1],"multi_ef_byteoffset.bin"),
         postings=mapped(argv[1],"postings_ef.bin");
    if(single.size!=singles*4 || counts.size!=(terms-singles)*4 || offsets.size!=(terms-singles)*8)
        fail("Index count/file size mismatch");
    void *library=dlopen("libcrypto.so.3",RTLD_NOW); if(!library) fail("OpenSSL 3 runtime missing");
    void *(*context_new)(void)=dlsym(library,"EVP_MD_CTX_new");
    void (*context_free)(void *)=dlsym(library,"EVP_MD_CTX_free");
    const void *(*sha256)(void)=dlsym(library,"EVP_sha256");
    int (*init)(void *,const void *,void *)=dlsym(library,"EVP_DigestInit_ex");
    int (*update)(void *,const void *,size_t)=dlsym(library,"EVP_DigestUpdate");
    int (*final)(void *,unsigned char *,unsigned *)=dlsym(library,"EVP_DigestFinal_ex");
    if(!context_new || !context_free || !sha256 || !init || !update || !final) fail("OpenSSL API missing");
    void *context=context_new(); if(!context) fail("Digest allocation failed");
    unsigned char xor_digest[32]={0},sum_digest[32]={0},digest[32];
    size_t position=0,previous_posting_end=0;
    for(uint64_t i=0;i<terms;i++) {
        unsigned char *term=vocabulary.data+position;
        unsigned char *end=memchr(term,0,vocabulary.size-position);
        if(!end || end==term) fail("Vocabulary malformed");
        uint64_t length=(uint64_t)(end-term); position+=length+1;
        uint32_t count=1; const unsigned char *payload; size_t bytes;
        if(i<singles) {
            payload=single.data+i*4; bytes=4;
            if(u32(payload)>=lines) fail("Singleton line ID outside corpus");
        } else {
            uint64_t j=i-singles,start=u64(offsets.data+j*8);
            uint64_t finish=j+1<terms-singles?u64(offsets.data+(j+1)*8):postings.size;
            count=u32(counts.data+j*4);
            if(count<2 || count>lines || start!=previous_posting_end || finish<start || finish>postings.size)
                fail("Posting offsets/counts malformed");
            bytes=(size_t)(finish-start);
            if(bytes!=encoded_size(count,lines)) fail("Noncanonical posting length");
            payload=postings.data+start; previous_posting_end=(size_t)finish;
        }
        unsigned digest_size=0;
        if(init(context,sha256(),NULL)!=1 || update(context,&length,8)!=1 ||
           update(context,term,(size_t)length)!=1 || update(context,&count,4)!=1 ||
           update(context,payload,bytes)!=1 || final(context,digest,&digest_size)!=1 || digest_size!=32)
            fail("SHA256 failed");
        unsigned carry=0;
        for(int k=31;k>=0;k--) { xor_digest[k]^=digest[k];
            unsigned value=sum_digest[k]+digest[k]+carry; sum_digest[k]=(unsigned char)value; carry=value>>8; }
    }
    if(position!=vocabulary.size || previous_posting_end!=postings.size) fail("Unconsumed index content");
    context_free(context);
    printf("{\"terms\":%"PRIu64",\"singletons\":%"PRIu64",\"lines\":%"PRIu64",\"sha256_xor\":\"",terms,singles,lines);
    hexadecimal(xor_digest); printf("\",\"sha256_sum\":\""); hexadecimal(sum_digest); printf("\"}\n");
    return 0;
}
