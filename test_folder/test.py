import ray
ray.init(address="auto")

@ray.remote
def f(x):
    return x * x

tasks = [f.remote(i) for i in range(10)]
print(ray.get(tasks))
