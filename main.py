from mrunner.helpers.client_helper import get_configuration

params = get_configuration()

print("learning rate:", params.learning_rate)
print("batch size:", params.batch_size)

# training code...
