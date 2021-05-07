import numpy as np
import pickle

class rating:
    
    def __init__(self):
        # weights for dimension
        self.dimension = [[0.21, 0.12, 0.25, 0.18, 0.12, 0.12]]
        # weights for dimension1's indicators
        self.indicator1 = [
            [1, 1, 1, 0.6, 0.4, 1, 1, 1, 1, 1],
            [0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.3, 0.3, 0.4],
            [0.29, 0.19, 0.23, 0.29]
        ]
        # weights for dimension2's indicators
        self.indicator2 = [
            [0.4, 0.6, 0.4, 0.6, 0.06, 0.25, 0.22, 0.19, 0.11, 0.12, 0.05, 1, 1, 1],
            [0.16, 0.25, 0.18, 0.13, 0.15, 0.13]
        ]
        # weights for dimension3's indicators
        self.indicator3 = [
            [0.3, 0.7, 0.34, 0.31, 0.35, 0.31, 0.31, 0.38],
            [0.41, 0.35, 0.24]
        ]
        # weights for dimension4's indicators
        self.indicator4 = [[0.25, 0.21, 0.15, 0.21, 0.18]]
        # weights for dimension5's indicators
        self.indicator5 = [[0.55, 0.45]]
        # weights for dimension6's indicators
        self.indicator6 = [[0.50, 0.50]]
        
        # combine weights together
        self.indicators = [
            self.indicator1, self.indicator2, self.indicator3, self.indicator4, self.indicator5, self.indicator6
        ]
        # calculate the number of factors for each dimension
        self.num_factors = [len(item[0]) for item in self.indicators]
    
    def preprocess(self, inputs_list_one_by_one):
        inputs, start = [], 0
        for num in self.num_factors:
            inputs.append(inputs_list_one_by_one[start : start+num])
            start += num
        return inputs
        
    def score(self, weights, num_input):
        w_sum = 0
        ws = []
        inputs = []
        scores = []
        for i, w in enumerate(weights):
            if w == 1:
                score = w * num_input[i]
            else:
                w_sum += w
                ws.append(w)
                inputs.append(num_input[i])
                if w_sum == 1:
                    score = np.sum(np.array(ws) * np.array(inputs))
                    w_sum = 0
                    ws = []
                    inputs = []
                else:
                    continue
            scores.append(score)
        return scores
    
    def score_dimension(self, dimension, num_input):
        for item in dimension:
            if item:
                score = self.score(item, num_input)
                num_input = score
        return round(score[0], 2)
    
    def predict(self, inputs):
        scores_dim = list(map(self.score_dimension, self.indicators, inputs))
        scores_overall = self.score_dimension(self.dimension, scores_dim)
        return scores_dim, scores_overall
    
def main():
    mod = rating()
    # save model to .pkl file
    with open("rating.pkl", "wb") as f:
        pickle.dump(mod, f)
    print("Model dumped!")
    
if __name__ == "__main__":
    main()
