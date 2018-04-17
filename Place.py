def Place_Algorithm(Physer, FlavorDict, Resource, Predict_number, ResourSet):
    Flavor_sort = dict({})
    for res in ResourSet:
        Flavor_sort[res] = []
    for res in ResourSet:
        for flavor in FlavorDict.keys():
            Flavor_sort[res].append(FlavorDict[flavor][res])
    print Flavor_sort
    
    result = []        
    return result