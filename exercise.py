# Jessica Turner, 6/11/2021

# known points for the function (x1, x2, y)
points = {(0, 1): 7, (1, 0): 5, (1, 1): 10}

# loss/cost
def J(w, b, c):

    loss = 0
    num_points = len(points)
    for input_pair in points:

        loss += (((w * input_pair[0]) + b + (c * input_pair[1]) - points[input_pair]) ** 2)

    # since J(w, b, c) = 1/N summation(y - y_true)
    return loss / num_points


# initial values for weights and biases
w_prev = 1
b_prev = 0
c_prev = 1

# learning rate
alpha = 0.45
# desired accuracy
epsilon = 0.0001

num_iterations = 100

# to store computations from gradient decent
w_new = 0
b_new = 0
c_new = 0

# train
for i in range(num_iterations):
    print('\n' + ("=" * 30) + " iteration {} ".format(i + 1) + ("=" * 30))

    dJ_dw = (2/3) * (2 * w_prev + 2 * b_prev + c_prev - 15)
    w_new = w_prev - alpha * dJ_dw

    dJ_db = (2/3) * (2 * w_prev + 3 * b_prev + 2 * c_prev - 22)
    b_new = b_prev - alpha * dJ_db

    dJ_dc = (2/3) * (w_prev + 2 * b_prev + 2 * c_prev - 17)
    c_new = c_prev - alpha * dJ_dc

    print('New vals:' + '\n\tw = {}\n\tb = {}\n\tc = {}'.format(w_new, b_new, c_new))
    current_loss_diff = abs(J(w_new, b_new, c_new) - J(w_prev, b_prev, c_prev))
    # check that its approaching 0 (where min occurs)
    print('current loss difference =', current_loss_diff)

    if current_loss_diff < epsilon:
        print("\nDesired accuracy reached !! Woop!")
        break

    # update vals
    w_prev = w_new
    b_prev = b_new
    c_prev = c_new