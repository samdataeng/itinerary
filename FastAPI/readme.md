with Docker :
- pull the image from DockerHub :
docker pull samiadata/itinerary_fastapi:1.2.0
- run the container :
docker run -p 8000:8000 samiadata/itinerary_fastapi:1.2.0
- go to :
`http://127.0.0.1:8000/docs`

Without Docker :
- Download the files on a project folder on your machine
- Python needs to be installed on your machine
- Install the requirements:
`pip install -r requirements.txt`
- inside the project folder, launch the API:
Linux : `uvicorn main:api --reload`
Windows : `python3 -m uvicorn main:api`
- go to :
`http://127.0.0.1:8000/docs`

