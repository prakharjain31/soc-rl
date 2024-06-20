import json
import numpy as np
import matplotlib.pyplot as plt

def inv_transform(distribution: str, num_samples: int, **kwargs) -> list:
    """ populate the 'samples' list from the desired distribution """

    samples = []

    # TODO: first generate random numbers from the uniform distribution
    if(distribution == "exponential"):
        samples = np.random.uniform(0,1,num_samples)
        samples = -1 * np.log(samples) / kwargs["lambda"]
        samples = samples.round(4)
        samples = np.ndarray.tolist(samples)
    else:
        samples = np.random.uniform(0,1,num_samples)
        samples = kwargs["gamma"] * np.tan(np.pi * (samples - 0.5)) + kwargs["peak_x"]
        samples = samples.round(4)
        samples = np.ndarray.tolist(samples)

    # END TODO
            
    return samples


if __name__ == "__main__":
    np.random.seed(42)

    for distribution in ["cauchy", "exponential"]:
        file_name = "./q1_" + distribution + ".json"
        args = json.load(open(file_name, "r"))
        samples = inv_transform(**args)
        
        with open("q1_output_" + distribution + ".json", "w") as file:
            json.dump(samples, file)

        # TODO: plot and save the histogram to "q1_" + distribution + ".png"
        plt.hist(samples , bins = 100)
        plt.savefig(f"q1_{distribution}.png")
        # END TODO
