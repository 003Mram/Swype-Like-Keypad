'''
You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.

You can also import packages you want.

But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.

NAME : Sriram Tirupattur    
ID : 112670605
DATE :
HOMEWORK :

'''

from flask import Flask, request
from flask import render_template
import time
import json
from scipy.interpolate import interp1d
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__)

centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

words, probabilities = [], {}
# For each word in the dictionary, it appends a list which involves the centroids of each character in that word.(2D list)
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])


def generate_sample_points(points_X, points_Y):
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''
    # TODO: Start sampling 

    distance = np.cumsum(np.sqrt( np.ediff1d(points_X, to_begin=0)**2 + np.ediff1d(points_Y, to_begin=0)**2 ))
    distance = distance/distance[-1]

    fx, fy = interp1d( distance, points_X), interp1d( distance, points_Y )

    alpha = np.linspace(0, 1, 100)
    sample_points_X, sample_points_Y = fx(alpha), fy(alpha)
    return sample_points_X, sample_points_Y



template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)



def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider reasonable)
    to narrow down the number of valid words so that ambiguity can be avoided.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    valid_words, valid_probabilities, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], [],[]
    # TODO: Set your own pruning threshold
    threshold = 35#Enter Value Here
    for i in range(10000):
        template_start = template_sample_points_X[i][0], template_sample_points_Y[i][0]
        template_end = template_sample_points_X[i][-1], template_sample_points_Y[i][-1]
        gesture_start = gesture_points_X[0], gesture_points_Y[0]
        gesture_end = gesture_points_X[-1], gesture_points_Y[-1]
        start_start_distance = math.sqrt((template_sample_points_X[i][0] - gesture_points_X[0])** 2 + (template_sample_points_Y[i][0] -gesture_points_Y[0])** 2 )
        end_end_distance = math.sqrt((template_sample_points_X[i][-1] - gesture_points_X[-1])**2 + (template_sample_points_Y[i][-1] - gesture_points_Y[-1])**2)
        if start_start_distance < threshold and end_end_distance < threshold:
            valid_template_sample_points_X.append(template_sample_points_X[i])
            valid_template_sample_points_Y.append(template_sample_points_Y[i])
            valid_words.append(words[i])
            valid_probabilities.append(probabilities[words[i]])

    # TODO: Do pruning 
    #return valid_probabilities
    return valid_words, valid_probabilities, valid_template_sample_points_X, valid_template_sample_points_Y

def get_scaled_points(sample_points_X, sample_points_Y, L):
    x_maximum = max(sample_points_X)
    x_minimum = min(sample_points_X)
    W = x_maximum - x_minimum
    y_maximum = max(sample_points_Y)
    y_minimum = min(sample_points_Y)
    H = y_maximum - y_minimum
    r = L/max(H, W,1)

    gesture_X, gesture_Y = [], []
    for point_x, point_y in zip(sample_points_X, sample_points_Y):
        gesture_X.append(r * point_x)
        gesture_Y.append(r * point_y)

    centroid_x = (max(gesture_X) - min(gesture_X))/2
    centroid_y = (max(gesture_Y) - min(gesture_Y))/2
    scaled_X, scaled_Y = [], []
    for point_x, point_y in zip(gesture_X, gesture_Y):
        scaled_X.append(point_x - centroid_x)
        scaled_Y.append(point_y - centroid_y)
    return scaled_X, scaled_Y

def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''
    # TODO: Set your own L
    L = 1
    # TODO: Calculate shape scores
    shape_scores = [] 
    scaled_X, scaled_Y= get_scaled_points(gesture_sample_points_X, gesture_sample_points_Y, L)
    valid_scaled_template_X, valid_scaled_template_Y = [], []
    for i in range(len(valid_template_sample_points_Y)):
        x, y = get_scaled_points(valid_template_sample_points_X[i], valid_template_sample_points_Y[i], L)
        valid_scaled_template_X.append(x)
        valid_scaled_template_Y.append(y)

    for i in range(len(valid_scaled_template_X)):
        shape = 0
        for j in range(len(valid_scaled_template_X[0])):
            shape += math.sqrt((valid_scaled_template_X[i][j] - scaled_X[j])**2  + (valid_template_sample_points_Y[i][j] - scaled_Y[j])**2)
        shape_scores.append(shape / 100)

    return shape_scores

def get_small_d(p_X, p_Y, q_X, q_Y):
    min_distance = []
    for n in range(0, 100):
        distance = math.sqrt((p_X - q_X[n])**2 + (p_Y - q_Y[n])**2)
        min_distance.append(distance)
    return (sorted(min_distance)[0])

def get_big_d(p_X, p_Y, q_X, q_Y, r):
    final_max = 0
    for n in range(0, 100):
        local_max = 0
        distance = get_small_d(p_X[n], p_Y[n], q_X, q_Y)
        local_max = max(distance-r , 0)
        final_max += local_max
    return final_max

def get_Ds(u_X, u_Y, t_X, t_Y, r):
    D1 = get_big_d(u_X, u_Y, t_X, t_Y, r)
    D2 = get_big_d(t_X, t_Y, u_X, u_Y, r)
    return D1, D2

def get_delta(u_X, u_Y, t_X, t_Y, r, i, D1, D2):
    if D1 == 0 and D2 == 0:
        return 0
    else:
        return math.sqrt((u_X[i] - t_X[i])**2 + (u_Y[i] - t_Y[i])**2)

def get_alphas():
    alphas = [0] * 100
    mid_point = 50
    for i in range(50):
        alphas[50- i - 1] = i / 2450
        alphas[50 + i] = i / 2450
    return alphas

def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''
    location_scores = []
    radius = 15
    # TODO: Calculate location scores
    alpha_scores = get_alphas() 
    delta_scores = []
    for i in range(len(valid_template_sample_points_X)):
        delta = []
        D1, D2 = get_Ds(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X[i], valid_template_sample_points_Y[i], radius)
        for j in range(100):
            delta.append(get_delta(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X[i], valid_template_sample_points_Y[i], radius, j, D1, D2))
        delta_scores.append(delta)

    for i in range(len(delta_scores)):
        lscore = 0
        for j in range(100):
            lscore += (alpha_scores[j] * delta_scores[i][j])
        location_scores.append(lscore)
    return location_scores


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.2#Enter Value Here#
    # TODO: Set your own location weight
    location_coef = 0.8 #Enter Value Here#
    for i in range(len(shape_scores)):
        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])
    return integration_scores


def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    best_word = ''
    # TODO: Set your own range.
    n =  5#Enter Value Here
    if not integration_scores: return 'No word exists. Try again.' 
    top_indices = sorted(range(len(integration_scores)), key=lambda i: integration_scores[i])[:n]
    top_words = []
    for idx in top_indices:
        top_words.append(valid_words[idx])

    print(top_words)
    weighted_integration = {}
    for idx in top_indices:
        weighted_integration[idx] = ( integration_scores[idx] / probabilities[valid_words[idx]])

    bestidx = min(weighted_integration, key= lambda x: weighted_integration[x]) 
    best_word = valid_words[bestidx]

    print(weighted_integration)
    # TODO: Get the best word 
    return top_words[0]


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():

    start_time = time.time()
    data = json.loads(request.get_data())
    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])
    gesture_points_X = [gesture_points_X]
    gesture_points_Y = [gesture_points_Y]

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X[-1], gesture_points_Y[-1])#Generate Sample Points

    valid_words, valid_probabilities, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_sample_points_X, gesture_sample_points_Y, template_sample_points_X, template_sample_points_Y) #Do Pruning

    shape_scores =get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)  #Get Shape Scores

    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)#Get Location Scores
    
    integration_scores = get_integration_scores(shape_scores, location_scores)#Get Integration Scores

    best_word = get_best_word(valid_words, integration_scores)#Get Best Word

    end_time = time.time()
    
    print('{"best_word": "' + best_word + '", "elapsed_time": "' + str(round((end_time - start_time) * 1000, 5)) + ' ms"}')

    return '{"best_word": "' + best_word + '", "elapsed_time": "' + str(round((end_time - start_time) * 1000, 5)) + ' ms"}'


if __name__ == "__main__":
    app.run()
