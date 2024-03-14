#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>

typedef struct _queue *queue;

// circular array
typedef struct _queue {
    int size;
    int used;
    int first;
    void **data;
    pthread_mutex_t mutex_queue;
    pthread_cond_t empty;
    pthread_cond_t full;
} _queue;

#include "queue.h"

queue q_create(int size) {
    queue q = malloc(sizeof(_queue));

    q->size  = size;
    q->used  = 0;
    q->first = 0;
    q->data  = malloc(size*sizeof(void *));

    pthread_mutex_init(&q->mutex_queue, NULL);
    pthread_cond_init(&q->empty, NULL);
    pthread_cond_init(&q->full, NULL);

    return q;
}

int q_elements(queue q) {
    return q->used;
}

int q_insert(queue q, void *elem) {

    pthread_mutex_lock(&q->mutex_queue);
    while (q->used == q->size) { //esta lleno
        pthread_cond_wait(&q->full, &q->mutex_queue);
    }

    q->data[(q->first+q->used) % q->size] = elem;
    q->used++;

    if (q->used == 1) {
        pthread_cond_broadcast(&q->empty);
    }
    pthread_mutex_unlock(&q->mutex_queue);

    return 1;
}

void *q_remove(queue q) {
    void *res;

    pthread_mutex_lock(&q->mutex_queue);
    while (!q->used) { //esta vacio
        pthread_cond_wait(&q->empty, &q->mutex_queue);
    }

    res = q->data[q->first];
    q->first = (q->first+1) % q->size;
    q->used--;

    if (q->used == q->size - 1) {
        pthread_cond_broadcast(&q->full);
    }
    pthread_mutex_unlock(&q->mutex_queue);

    return res;
}

void q_destroy(queue q) {
    free(q->data);
    free(q);
}
