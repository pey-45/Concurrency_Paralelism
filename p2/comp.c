#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <pthread.h>
#include "compress.h"
#include "chunk_archive.h"
#include "queue.h"
#include "options.h"

#define CHUNK_SIZE (1024*1024)
#define QUEUE_SIZE 20

typedef struct shared {
    queue in, out;
    archive ar;
    int chunks, fd;
    struct options *opt;
} shared_t;

// take chunks from queue in, run them through process (compress or decompress), send them to queue out
void* worker(void * ptr) {
    shared_t *shared = ptr;
    chunk ch, res;

    while(q_elements(shared->in) > 0) {
        ch = q_remove(shared->in);

        res = zcompress(ch);
        free_chunk(ch);

        q_insert(shared->out, res);
    }

    return (void*)0;
}

void* readChunks(void * ptr) {
    shared_t *shared = ptr;
    chunk ch;
    int i, offset;

    for(i=0; i<shared->chunks; i++) {
        ch = alloc_chunk(shared->opt->size);

        offset = (int)lseek(shared->fd, 0, SEEK_CUR);

        ch->size = (int)read(shared->fd, ch->data, shared->opt->size);
        ch->num = i;
        ch->offset = offset;

        q_insert(shared->in, ch);
    }

    return (void*)0;
}

void* writeChunks(void * ptr) {
    shared_t *shared = ptr;
    chunk ch;
    int i;

    for(i=0; i<shared->chunks; i++) {
        ch = q_remove(shared->out);

        add_chunk(shared->ar, ch);
        free_chunk(ch);
    }

    return (void*)0;
}

void startThreads(shared_t * shared, struct options opt) {
    pthread_t threads[opt.num_threads]; //hilos para worker
    pthread_t thread_read; //hilo para lectura de chunks
    pthread_t thread_write; //hilo para escritura de chunks
    int i;
    
    pthread_create(&thread_read, NULL, readChunks, shared);
    for (i = 0; i < opt.num_threads; i++) {
        pthread_create(&threads[i], NULL, worker, shared);
    }
    pthread_create(&thread_write, NULL, writeChunks, shared);

    pthread_join(thread_read, NULL);
    for (i = 0; i < opt.num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    pthread_join(thread_write, NULL);
}

// Compress file taking chunks of opt.size from the input file,
// inserting them into the in queue, running them using a worker,
// and sending the output from the out queue into the archive file
void comp(struct options opt) {
    char comp_file[256];
    struct stat st;
    shared_t shared;

    if((shared.fd=open(opt.file, O_RDONLY))==-1) {
        printf("Cannot open %s\n", opt.file);
        exit(0);
    }

    fstat(shared.fd, &st);
    shared.chunks = (int)st.st_size/opt.size+(st.st_size % opt.size ? 1:0);

    if(opt.out_file) {
        strncpy(comp_file,opt.out_file,255);
    } else {
        strncpy(comp_file, opt.file, 255);
        strncat(comp_file, ".ch", 255);
    }

    shared.ar = create_archive_file(comp_file);
    shared.in  = q_create(opt.queue_size);
    shared.out = q_create(opt.queue_size);
    shared.opt = &opt;

    startThreads(&shared, opt);

    close_archive_file(shared.ar);
    close(shared.fd);
    q_destroy(shared.in);
    q_destroy(shared.out);
}


// Decompress file taking chunks of size opt.size from the input file

void decomp(struct options opt) {
    int fd, i;
    char uncomp_file[256];
    archive ar;
    chunk ch, res;

    if((ar=open_archive_file(opt.file))==NULL) {
        printf("Cannot open archive file\n");
        exit(0);
    }

    if(opt.out_file) {
        strncpy(uncomp_file, opt.out_file, 255);
    } else {
        strncpy(uncomp_file, opt.file, strlen(opt.file) -3);
        uncomp_file[strlen(opt.file)-3] = '\0';
    }

    if((fd=open(uncomp_file, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH))== -1) {
        printf("Cannot create %s: %s\n", uncomp_file, strerror(errno));
        exit(0);
    }

    for(i=0; i<chunks(ar); i++) {
        ch = get_chunk(ar, i);

        res = zdecompress(ch);
        free_chunk(ch);

        lseek(fd, res->offset, SEEK_SET);
        write(fd, res->data, res->size);
        free_chunk(res);
    }

    close_archive_file(ar);
    close(fd);
}

int main(int argc, char *argv[]) {
    struct options opt;

    opt.compress    = 1;
    opt.num_threads = 3;
    opt.size        = CHUNK_SIZE;
    opt.queue_size  = QUEUE_SIZE;
    opt.out_file    = NULL;

    read_options(argc, argv, &opt);

    if(opt.compress) comp(opt);
    else decomp(opt);
}
