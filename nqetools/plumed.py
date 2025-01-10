def write_plumed_input(temperature=300, sigma=[0.005,0.05], ):
    impt = """# default units are LENGTH=nm ENERGY=kJ/mol TIME=ps\n"""

    restraints = {}

    restraints["doo"] = "DISTANCE ATOMS=1,2"
    restraints["co1"] = "DISTANCES GROUPA=1 GROUPB=3-7 LESS_THAN={RATIONAL R_0=0.14}"
    restraints["co2"] = "DISTANCES GROUPA=2 GROUPB=3-7 LESS_THAN={RATIONAL R_0=0.14}"
    restraints["dc"] = "COMBINE ARG=co1.lessthan,co2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO"

    # Iterate over the restraints and add them to the input file
    for key, value in restraints.items():
        impt += f"{key}: {value}\n"

    restraints_keys = list(restraints.keys())
    # convert the keys to a string
    restraints_str = ",".join(restraints_keys)

    # mtd line
    mtd_line = f"mtd: METAD ARG={restraints_str} PACE=10 \n"
    # add the mtd line to the input file
    impt += mtd_line
    #

    return impt