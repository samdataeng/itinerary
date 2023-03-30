With Docker : <br />
1- pull the image from DockerHub : <br />
`docker pull samiadata/itinerary_dash:1.2.0` <br />
2- run the container : <br />
`docker run -p 8050:8050 samiadata/itinerary_dash:1.2.0` <br />
3- go to : <br />
`http://localhost:8050` <br />

Without Docker (Python need to be installed on your machine) :<br />
1- dowload the Dash folder on your machine <br />
2 - Install the requirements: <br />
`pip install -r requirements.txt` <br />
3 - from the Dash folder, launch the app: <br />
`python3 app.py` <br />
5- go to : <br />
`http://localhost:8050`
