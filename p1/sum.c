#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "options.h"

struct nums {
    long *increase, *decrease, total, iterations_decr_incr, iterations_desplaz_incr, iterations_desplaz_decr;
    pthread_mutex_t mutex, *mutex_decr, *mutex_incr;
    int size;
};

struct args {
    int thread_num;		// application defined thread #
    long iterations;	// number of operations
    struct nums *nums;	// pointer to the counters (shared with other threads)
};

struct thread_info {
    pthread_t    id;    // id returned by pthread_create()
    struct args *args;  // pointer to the arguments
};

/*aux*/
long sum(long * v, int n) {
    int i;
    long sum = 0;
    for (i = 0; i < n; i++) {
        sum += v[i];
    }
    return sum;
}

/*aux*/
void printArray(long * v, int n) {
    int i;
    printf("[ ");
    for (i = 0; i < n; i++) {
        printf("%ld ", v[i]);
    }
    printf("]");
}

void lockMutex(pthread_mutex_t * mutex, pthread_mutex_t * mutex_d, pthread_mutex_t * mutex_i) {
    pthread_mutex_lock(mutex);
    pthread_mutex_lock(mutex_d);
    pthread_mutex_lock(mutex_i);
}

void unlockMutex(pthread_mutex_t * mutex_d, pthread_mutex_t * mutex_i, pthread_mutex_t * mutex) {
    pthread_mutex_unlock(mutex_d);
    pthread_mutex_unlock(mutex_i);
    pthread_mutex_unlock(mutex);
}

void *decrease_increase(void * ptr)
{
    struct args *args = ptr;
    struct nums *n = args->nums;
    int nd, ni;

    while(1) {
        /* obtenemos una posicion random para cada array,
         * son privados para cada hilo, por tanto, no es necesario protegerlos*/
        nd = rand()%n->size;
        ni = rand()%n->size;

        lockMutex(&n->mutex, &n->mutex_decr[nd], &n->mutex_incr[ni]); //bloquea los mutex

        if (!n->iterations_decr_incr) {
            unlockMutex(&n->mutex_decr[nd], &n->mutex_incr[ni], &n->mutex); //desbloquea los mutex
            break;
        } else {
            n->iterations_decr_incr--;
        }

        n->decrease[nd]--;
        n->increase[ni]++;

        unlockMutex(&n->mutex_decr[nd], &n->mutex_incr[ni], &n->mutex); //desbloquea los mutex
    }
    return NULL;
}

/*funcion que ejecutaran los nuevos tipos de hilo*/
void *desplaz_increment(void * ptr) {
    struct args *args = ptr;
    struct nums *n = args->nums;
    int ni1, ni2;

    while(1) {
        /* obtenemos dos posicion random del array de incrementos,
         * son privados para cada hilo, por tanto, no es necesario protegerlos*/
        ni1 = rand()%n->size;
        ni2 = rand()%n->size;

        lockMutex(&n->mutex, &n->mutex_decr[ni1], &n->mutex_incr[ni2]); //bloquea los mutex

        /*esta sentencia purga la cola de intentos de acceso a la zona critica si se completaron todas las iteraciones*/
        if (!n->iterations_desplaz_incr) {
            unlockMutex(&n->mutex_decr[ni1], &n->mutex_incr[ni2], &n->mutex); //desbloquea los mutex
            break;
        } else {
            n->iterations_desplaz_incr--;
        }

        n->increase[ni1]--;
        n->increase[ni2]++;

        unlockMutex(&n->mutex_decr[ni1], &n->mutex_incr[ni2], &n->mutex); //desbloquea los mutex
    }
    return NULL;
}

/*funcion que ejecutaran los nuevos tipos de hilo*/
void *desplaz_decrement(void * ptr) {
    struct args *args = ptr;
    struct nums *n = args->nums;
    int nd1, nd2;

    while(1) {
        /* obtenemos dos posiciones random del array de decrenentos,
         * son privados para cada hilo, por tanto, no es necesario protegerlos*/
        nd1 = rand()%n->size;
        nd2 = rand()%n->size;

        lockMutex(&n->mutex, &n->mutex_decr[nd1], &n->mutex_incr[nd2]); //bloquea los mutex

        /*esta sentencia purga la cola de intentos de acceso a la zona critica si se completaron todas las iteraciones*/
        if (!n->iterations_desplaz_decr) {
            unlockMutex(&n->mutex_decr[nd1], &n->mutex_incr[nd2], &n->mutex); //desbloquea los mutex
            break;
        } else {
            n->iterations_desplaz_decr--;
        }

        n->decrease[nd1]--;
        n->decrease[nd2]++;

        unlockMutex(&n->mutex_decr[nd1], &n->mutex_incr[nd2], &n->mutex); //desbloquea los mutex
    }
    return NULL;
}

/*funcion pasada como parámetro*/
struct thread_info *start_threads_heap(struct options opt, struct nums *nums, void * (*fun)(void*)) {
    int i;
    struct thread_info *threads;

    printf("creating %d threads\n", opt.num_threads);
    threads = malloc(sizeof(struct thread_info) * opt.num_threads);

    if (!threads) {
        printf("Not enough memory\n");
        exit(1);
    }

    for (i = 0; i < opt.num_threads; i++) {
        threads[i].args = malloc(sizeof(struct args));

        threads[i].args->thread_num = i;
        threads[i].args->nums       = nums;
        threads[i].args->iterations = opt.iterations;

        if (pthread_create(&threads[i].id, NULL, fun, threads[i].args)) {
            printf("Could not create thread #%d", i);
            exit(1);
        }
    }

    return threads;
}

struct thread_info *start_threads_stack(struct options opt, struct nums *nums, void *(*fun)(void*)) {
    int i;
    struct thread_info *threads;
    struct args args; //creacion de una variable tipo struct args en el stack del thread inicial

    printf("creating %d threads\n", opt.num_threads);
    threads = malloc(sizeof(struct thread_info) * opt.num_threads);

    if (!threads) {
        printf("Not enough memory\n");
        exit(1);
    }

    // Create num_thread threads running decrease_increase
    for (i = 0; i < opt.num_threads; i++) {
        //asignacion de valores al struct args del stack
        args.thread_num = i;
        args.nums       = nums;
        args.iterations = opt.iterations;

        //se le asignan los argumentos del stack a los argumentos del thread de la iteracion actual
        threads[i].args = &args;

        if (pthread_create(&threads[i].id, NULL, fun, threads[i].args)) {
            printf("Could not create thread #%d", i);
            exit(1);
        }
    }

    return threads;
}

void print_totals(struct nums * nums) {
    printf("Final: increasing %ld decreasing %ld diff %ld\n",
           sum(nums->increase, nums->size), sum(nums->decrease, nums->size),
           nums->total*nums->size - sum(nums->increase, nums->size) - sum(nums->decrease, nums->size));
}

void wait_heap(struct options opt, struct thread_info * threads) {
    for (int i = 0; i < opt.num_threads; i++) {
        pthread_join(threads[i].id, NULL);
    }

    for (int i = 0; i < opt.num_threads; i++) {
        free(threads[i].args);
    }
}

void wait_stack(struct options opt, struct thread_info * threads) {
    for (int i = 0; i < opt.num_threads; i++) {
        pthread_join(threads[i].id, NULL);
    }
}

int main (int argc, char ** argv) {
    struct options opt;
    struct nums nums;
    struct thread_info *thrs_decr_incr, *thrs_desplaz_incr, *thrs_desplaz_decr; //creacion de nuevos tipos de hilo
    int i;

    srand(time(NULL));

    // Default values for the options
    opt.num_threads  = 4;
    opt.iterations   = 100000;
    opt.size         = 10;

    read_options(argc, argv, &opt);

    nums.total = opt.iterations;
    nums.iterations_decr_incr = opt.iterations;
    nums.iterations_desplaz_incr = opt.iterations;
    nums.iterations_desplaz_decr = opt.iterations;
    nums.increase = malloc(opt.size*sizeof(long));
    nums.decrease = malloc(opt.size*sizeof(long));
    nums.mutex_decr = malloc(opt.size*sizeof(pthread_mutex_t));
    nums.mutex_incr = malloc(opt.size*sizeof(pthread_mutex_t));
    nums.size = opt.size;

    for (i = 0; i < opt.size; i++) {
        nums.increase[i] = 0;
        nums.decrease[i] = nums.total;
        pthread_mutex_init(&nums.mutex_decr[i], NULL);
        pthread_mutex_init(&nums.mutex_incr[i], NULL);
    }

    pthread_mutex_init(&nums.mutex, NULL);

    thrs_decr_incr = start_threads_heap(opt, &nums, decrease_increase);
    thrs_desplaz_incr = start_threads_stack(opt, &nums, desplaz_increment);
    thrs_desplaz_decr = start_threads_heap(opt, &nums, desplaz_decrement);
    wait_heap(opt, thrs_decr_incr);
    wait_stack(opt, thrs_desplaz_incr);
    wait_heap(opt, thrs_desplaz_decr);

    for (i = 0; i < opt.size; i++) {
        pthread_mutex_destroy(&nums.mutex_decr[i]);
        pthread_mutex_destroy(&nums.mutex_incr[i]);
    }

    pthread_mutex_destroy(&nums.mutex);

    print_totals(&nums);

    free(nums.increase);
    free(nums.decrease);
    free(nums.mutex_incr);
    free(nums.mutex_decr);
    free(thrs_decr_incr);
    free(thrs_desplaz_incr);
    free(thrs_desplaz_decr);

    return 0;
}
