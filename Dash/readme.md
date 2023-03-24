
On your local machine : 
enter the folder where the files are (the python file must be named app.py)

Withour Docker :
- Install the requirements:
`pip install -r requirements.txt`
- create an 'assets' folder and put "la_reunion.jpg" in it
4- Launch the app:
'python3 app.py' 
5- go to :
`http://localhost:8050`

With Docker :

- create the image :
docker image build . -t itinerary_dash
- run the container :
docker docker run -p 8050:8050 itinerary_dash
- go to :
`http://localhost:8050`
