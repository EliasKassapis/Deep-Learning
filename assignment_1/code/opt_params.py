import numpy as np



def get_setups():


    l_size_options = np.linspace(100,1000,10).astype(int).astype(str)
    etas = np.linspace(1e-8, 1e-5, 4)
    opts = ["adam", "sgd"]
    batch_size = np.linspace(32,128, 4).astype(int)
    # n_hlayers = int(np.random.uniform(1,4))
    n_hlayers = np.linspace(1,4,4).astype(int)
    # l_size = np.random.choice(l_size_options)

    setups = []

    #create all combos
    for i in range(3):
        for n in n_hlayers:
            n_hidden = ''
            for layer in range(n):
                l_size = np.random.choice(l_size_options)
                if layer != (n-1):
                    n_hidden += l_size + ','
                else:
                    n_hidden += l_size
            for b_size in batch_size:
                for eta in etas:
                    for opt in opts:
                        setups.append([n_hidden, b_size, eta, opt])

    return setups

setups = get_setups()

print(len(setups))

