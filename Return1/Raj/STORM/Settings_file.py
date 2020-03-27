import json
import STORM_Samurai as sam

# CONVERTING JSON TO PYTHON:
# some JSON:
x = '{ "filename": "/Users/RajSeehra/University/Masters/Semester 2/Teaching_python-master/Week 2/Data/640.tif", "filter":"kernel"}'

# parse x:
y = json.loads(x)

# the result is a Python dictionary:
print(y["filename"])

# CONVERTING JSON TO PYTHON:


# CONVERTING PYTHON TO JSON:
# a Python object (dict):
x = {
  "filename": "/Users/RajSeehra/University/Masters/Semester 2/Teaching_python-master/Week 2/Data/640.tif",
  "filter": "kernel",
}

# convert into JSON:
y = json.dumps(x)

# the result is a JSON string:
print(y)

# Saving with json.
with open("data_file.json", "w") as write_file:
    json.dump(x, write_file)
# CONVERTING PYTHON TO JSON:


with open("data_file.json", "r") as read_file:
    data = json.load(read_file)

print(data["filename"])

x = sam.filter_switcher(data["filename"], data["filter"])
print(x)