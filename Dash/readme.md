With Docker :
- pull the image from DockerHub :
docker pull samiadata/itinerary_dash:1.2.0
- run the container :
docker run -p 8050:8050 samiadata/itinerary_dash:1.2.0
- go to :
`http://localhost:8050`

Without Docker :
-dowload the files on a project folder on your machine
- Python need to be installed on your machine
- Install the requirements:
`pip install -r requirements.txt`
- create an 'assets' folder and put "la_reunion.jpg" in it
4- from the project folder, launch the app:
'python3 app.py' 
5- go to :
`http://localhost:8050`


