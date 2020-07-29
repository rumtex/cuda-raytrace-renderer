/* Chemistry is the scientific discipline involved with elements and compounds composed of
** atoms, molecules and ions: their composition, structure, properties, behavior and the
** changes they undergo during a reaction with other substances
** (c) Wiki
*/

typedef struct {
    fract_t     radius;
    fract_t     weight;
} atom_t;

// Atom fraction
typedef struct {

} afract_t;

typedef struct {
    omg_t            atoms;
} material_t;

typedef struct {
    atom_t           atom;
    material_t       material;
} object_t;
