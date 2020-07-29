#define DEFAULT_OMG_BLOCK_SIZE 1000

typedef struct {
    unsigned                total;
    unsigned                count;
    void*                   items;
    void*                   prev_block;
    void*                   next_block;
} block_t;

typedef struct {
    unsigned                blocks;
    unsigned                item_size;
    block_t*                first_block;
} omg_t;

CROSS_COMPILING_OPTS omg_t init_omg(unsigned obj_size);
CROSS_COMPILING_OPTS omg_t init_omg_of(unsigned size, unsigned obj_size); // книжка с заранее известным количеством элементов
CROSS_COMPILING_OPTS void clear_omg(omg_t*);

CROSS_COMPILING_OPTS void add_block(omg_t*);
CROSS_COMPILING_OPTS void* new_omg_item(omg_t*);
