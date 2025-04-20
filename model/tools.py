from utils.CalculatedMaterials import CalculatedMaterials
def fit_composites(
    build_composites: list,
    set_thickness: list,
    wl: list,
    number_polyfit: list,
    method
):
    my_fit=CalculatedMaterials(
        build_composites=build_composites,
        set_thickness=set_thickness,
        wl=wl,
    )
