ObservationI: 

def get_delta(u_X, u_Y, t_X, t_Y, r, i):
    D1 = get_big_d(u_X, u_Y, t_X, t_Y, r)
    D2 = get_big_d(t_X, t_Y, u_X, u_Y, r)
    if D1 == 0 and D2 == 0:
        return 0
    else:
        return math.sqrt((u_X[i] - t_X[i])**2 + (u_Y[i] - t_Y[i])**2)


So, we call get_delta 100 times for 100 points (i)for each template but it you notice the get_delta,
The D1, D2 are called every single time. However, D1,D2 don't depend on i for a given template.
So, you can calculate D1,D2 only once for each template and store them.
This reduces the time from 300s to a few seconds.

Observation II:

def generate_sample_points(points_X, points_Y):

    distance = np.cumsum(np.sqrt( np.ediff1d(points_X, to_begin=0)**2 + np.ediff1d(points_Y, to_begin=0)**2 ))
    distance = distance/distance[-1]

    fx, fy = interp1d( distance, points_X), interp1d( distance, points_Y )

    alpha = np.linspace(0, 1, 100)
    sample_points_X, sample_points_Y = fx(alpha), fy(alpha)
    return sample_points_X, sample_points_Y

While generating 100 samples, we have to take into consideration the number of samples depending upon the distance between 2 points.

1->2->3 . this will have 50 samples between 1->2 and 50 samples between 2->3
1->2->9. This shoud have 10 samples between 1->2 and 70 samples between 2->9 for better results.


Observation III:

While taking the best word, we need to consider the least possible score of all as all the scores are calculated based on distance. So, the one with less score is the best suitable word.


Observation IV:

When I tried gesturing for Plot, i got put. I am gettting this because i am multiplying by the probability. Therefore, I am dividing by the probability scores so that integration score becomes even lesser and I'll select that word.


References:
1. https://stackoverflow.com/questions/51512197/python-equidistant-points-along-a-line-joining-set-of-points/51515357
2. https://github.com/rmodi6/gesture-recognition/blob/master/server.py

