import timeit
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import numpy as np
import copy
import matplotlib.pyplot as plt


def lmform(
    data_list,
    config,
    join,
    transfer=None):
    
    start = timeit.default_timer()
    
    np.random.seed(seed=config["seed"])
    
    convergence_parameters = {}
    convergence_parameters["count"] = 0
    convergence_parameters["score_vec"] = [10e6]
    
    main_parameters, main_code = initialise_lmform(data_list = data_list,
                                         config = config,
                                         join = join,
                                         transfer = transfer
                                        )
    
    if (config["verbose"]):
            print("Beginning lmform learning with:    Sample dimension reduction (config[i_dim]): " + str( config["i_dim"] ) + "    Feature dimension reduction (config[j_dim]): " + str( config["j_dim"] ) + "    Tolerance Threshold: " + str( config["tol"] ) + "   Maximum number of iterations: "  + str( config["max_iter"] ) + "   Verbose: ", config["verbose"])

            
    while True:
        prev_encode = main_code["encode"]
        
        for i in range(len(join["complete"]["data_list"])):
            internal_parameters = {}
            internal_parameters["beta"] = main_parameters["beta"][join["complete"]["beta"][i]]
            internal_parameters["intercept"] = main_parameters["intercept"][join["complete"]["data_list"][i]]
            
            internal_code = {}
            internal_code["encode"] = main_code["encode"][join["complete"]["code"][i]]
            internal_code["code"] = main_code["code"][join["complete"]["code"][i]]
            
            return_parameters, return_code  = update_set_lmform( 
                                        x = data_list[join["complete"]["data_list"][i]],
                                        main_parameters = internal_parameters,
                                        main_code = internal_code,
                                        config = config,
                                        fix = transfer["fix"]
                                        )

            main_parameters["beta"][join["complete"]["beta"][i]] = internal_parameters["beta"]
            main_parameters["intercept"][join["complete"]["data_list"][i]] = internal_parameters["intercept"]
            
            main_code["code"][join["complete"]["code"][i]] = internal_code["code"]
            main_code["encode"][join["complete"]["code"][i]] = internal_code["encode"]
            
        total_mae = 0
        for X in range(len(join["complete"]["data_list"])):      
            total_mae += torch.mean(torch.abs(main_code["encode"][join["complete"]["code"][X]] - prev_encode[join["complete"]["code"][X]]))

        # Check convergence
        convergence_parameters["score_vec"] += [total_mae]
        MSE = convergence_parameters["score_vec"][-1]
        prev_MSE = convergence_parameters["score_vec"][-2]
        
        if convergence_parameters["count"]>=1:
            if config["verbose"]:
                print("Iteration:   "+str(convergence_parameters["count"])+"   with Tolerance of:   "+str(abs(prev_MSE - MSE)))
            if convergence_parameters["count"] >= config["max_iter"]:
                break
            if abs(prev_MSE - MSE) < config["tol"]:
                break
        convergence_parameters["count"] += 1

    if (config["verbose"]):
        print("Learning has converged for lmform, beginning (if requested) dimension reduction")

    return_data = {}
    return_data["main_parameters"] = main_parameters
    return_data["main_code"] = main_code
    return_data["meta_parameters"] = {}
    return_data["meta_parameters"]["config"] = config
    return_data["meta_parameters"]["join"] = join
    return_data["convergence_parameters"] = convergence_parameters
    
    stop = timeit.default_timer()

    return_data["run_time"] = {}
    return_data["run_time"]["start"] = start
    return_data["run_time"]["stop"] = stop
    return_data["run_time"]["run_time"] = stop - start
    
    return return_data
               
               
def initialise_lmform(
    data_list,
    config,
    join,
    transfer
):

    main_code = {}
    main_code["code"] = {}
    main_code["encode"] = {}

    main_parameters = {}
    main_parameters["beta"] = {}
    main_parameters["intercept"] = {}
    
    for i in range(len(join["complete"]["data_list"])):
        main_code["code"][join["complete"]["code"][i]] = []
        main_code["encode"][join["complete"]["code"][i]] = []

        main_parameters["beta"][join["complete"]["beta"][i]] = []
        main_parameters["intercept"][join["complete"]["data_list"][i]] = []
    

    for i in range(len(join["complete"]["data_list"])):

        if main_parameters["beta"][join["complete"]["beta"][i]] == []:
            if not len(transfer["main_parameters"]["beta"][join["complete"]["beta"][i]]) == 0:
                main_parameters["beta"][join["complete"]["beta"][i]] = transfer["main_parameters"]["beta"][join["complete"]["beta"][i]]
            else:
                main_parameters["beta"][join["complete"]["beta"][i]] = initialise_parameters_lmform(x = data_list[join["complete"]["data_list"][i]], dim_main = config["j_dim"], seed_main = 1, type_main = config["init"]["beta"]).T

        if main_code["code"][join["complete"]["encode"][i]] == []:
            if not len(transfer["main_code"]["encode"][join["complete"]["encode"][i]]) == 0:
                main_code["encode"][join["complete"]["code"][i]] = transfer["main_code"]["encode"][join["complete"]["code"][i]]
            else:
                main_code["encode"][join["complete"]["code"][i]] = data_list[join["complete"]["data_list"][i]]@main_parameters["beta"][join["complete"]["beta"][i]]

        if main_code["code"][join["complete"]["code"][i]] == []:
            if not len(transfer["main_code"]["code"][join["complete"]["code"][i]]) == 0:
                main_code["code"][join["complete"]["code"][i]] = transfer["main_code"]["code"][join["complete"]["code"][i]]
            else:
                main_code["code"][join["complete"]["code"][i]] = data_list[join["complete"]["data_list"][i]]@torch.linalg.pinv((main_parameters["beta"][join["complete"]["beta"][i]]).T)

        if main_parameters["intercept"][join["complete"]["data_list"][i]] == []:
            if not len(transfer["main_parameters"]["intercept"][join["complete"]["data_list"][i]]) == 0:
                main_parameters["intercept"][join["complete"]["data_list"][i]] = transfer["main_parameters"]["intercept"][join["complete"]["data_list"][i]]
            else:
                main_parameters["intercept"][join["complete"]["data_list"][i]] = torch.mean(data_list[join["complete"]["data_list"][i]] - main_code["code"][join["complete"]["code"][i]]@(main_parameters["beta"][join["complete"]["beta"][i]].T),0)
        
    return main_parameters, main_code
               
               
               
def initialise_parameters_lmform(
                            x,
                            dim_main, 
                            seed_main,
                            type_main):
    if type_main == "SVD":
        svd = TruncatedSVD(n_components=dim_main, n_iter=2, random_state=seed_main)
        return svd.fit(x).components_

    
    if type_main == "rand":
        rand_data = np.random.randn(dim_main,x.shape[1])
        return rand_data
    
    
               
def update_set_lmform(x,main_parameters,main_code,config,fix):

    if not fix["code"]:
        main_code["code"] = (x - main_parameters["intercept"])@torch.linalg.pinv(main_parameters["beta"].T)
               
    if not fix["beta"]:
        main_parameters["beta"] = (torch.linalg.pinv(main_code["code"])@(x - main_parameters["intercept"])).T
               
    if not fix["intercept"]:
        main_parameters["intercept"] = torch.mean(x - (main_code["code"])@(main_parameters["beta"]).T,0)
    
    if not fix["encode"]:
        main_code["encode"] = (x - main_parameters["intercept"])@(main_parameters["beta"])

    return main_parameters, main_code