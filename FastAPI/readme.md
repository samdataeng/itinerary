with Docker : <br />
- pull the image from DockerHub : <br />
`docker pull samiadata/itinerary_fastapi:1.2.0` <br />
- run the container : <br />
`docker run -p 8000:8000 samiadata/itinerary_fastapi:1.2.0` <br />
- go to : <br />
`http://127.0.0.1:8000/docs`

Without Docker (Python needs to be installed on your machine) : <br />
- Download the FastApi folder on your machine <br />
- Install the requirements: <br />
`pip install -r requirements.txt` <br />
- from inside the FastAPI folder, launch the API: <br />
Linux : `uvicorn main:api --reload` <br />
Windows : `python3 -m uvicorn main:api` <br />
- go to : <br />
`http://127.0.0.1:8000/docs`

