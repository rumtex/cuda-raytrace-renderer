#include <mapping/map.h>

CROSS_COMPILING_OPTS map_t init_map()
{
    map_t map;
    // init_const();

    map.atoms = init_omg(sizeof(atom_t));
    map.materials = init_omg(sizeof(material_t));
    map.objects = init_omg(sizeof(object_t));

    map.has_geometry = true;
    map.geometry = init_omg(sizeof(geometry_t));

    return map;
}

CROSS_COMPILING_OPTS void clear_map(map_t* map)
{
    clear_omg(&map->atoms);
    clear_omg(&map->materials);
    clear_omg(&map->objects);
    clear_omg(&map->geometry);
}

CROSS_COMPILING_OPTS void add_atom(map_t* map, atom_t a)
{
    *((atom_t*)new_omg_item(&map->atoms)) = a;
}

CROSS_COMPILING_OPTS void add_material(map_t* map, material_t m)
{
    *((material_t*)new_omg_item(&map->materials)) = m;
}

CROSS_COMPILING_OPTS void add_object(map_t* map, object_t o)
{
    *((object_t*)new_omg_item(&map->objects)) = o;
}

CROSS_COMPILING_OPTS void add_geometry(map_t* map, geometry_t g)
{
    *((geometry_t*)new_omg_item(&map->geometry)) = g;
}
