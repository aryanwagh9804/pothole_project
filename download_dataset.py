from roboflow import Roboflow

# paste your API key here
rf = Roboflow(api_key="6m9r7QdG66#####")

project = rf.workspace("vision-pqcbd").project("pothole-iqavm")
dataset = project.version(1).download("yolov8")

print("Dataset downloaded successfully")