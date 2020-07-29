#include <mapping/map.h>

CROSS_COMPILING_OPTS block_t* init_block(unsigned size, unsigned item_size)
{
    block_t* block = (block_t*) malloc(sizeof(block_t));

    block->total = size;
    block->count = 0;
    block->items = (void*) malloc(size * item_size);

    return block;
}

CROSS_COMPILING_OPTS omg_t init_omg(unsigned item_size)
{
    return init_omg_of(DEFAULT_OMG_BLOCK_SIZE, item_size);
}

CROSS_COMPILING_OPTS omg_t init_omg_of(unsigned size, unsigned item_size)
{
    omg_t omg;
    omg.item_size = item_size;
    omg.first_block = init_block(size, item_size);
    omg.first_block->prev_block = NULL;
    omg.first_block->next_block = NULL;
    omg.blocks = 1;
    return omg;
}

CROSS_COMPILING_OPTS void clear_omg(omg_t* omg)
{
    unsigned blocks_it = omg->blocks;
    while(blocks_it-- != 0) {
        free(omg->first_block->items);
        if (omg->first_block->next_block)
        {
            omg->first_block = (block_t*) omg->first_block->next_block;
            free(omg->first_block->prev_block);
        } else {
            free(omg->first_block);
        }
    }
}

CROSS_COMPILING_OPTS void add_block(omg_t* omg)
{
    unsigned blocks_it = omg->blocks;
    block_t* last_block = omg->first_block;
    while(--blocks_it != 0) {
        last_block = (block_t*) last_block->next_block;
    }
    last_block->next_block = (void*) init_block(DEFAULT_OMG_BLOCK_SIZE, omg->item_size);
    ((block_t*)last_block->next_block)->prev_block = last_block;
    omg->blocks++;
}

CROSS_COMPILING_OPTS void* new_omg_item(omg_t* omg)
{
    unsigned blocks_it = omg->blocks;
    block_t* last_block = omg->first_block;
    while(--blocks_it != 0) {
        last_block = (block_t*) last_block->next_block;
    }
    if (last_block->count == last_block->total) {
        add_block(omg);
        last_block = (block_t*) last_block->next_block;
    }

    return last_block->items + (omg->item_size * (last_block->count++));
}
