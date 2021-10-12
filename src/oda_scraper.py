import requests
import json

def scrape_odaData(ressource):

    content = []
    done = False
    url = "https://oda.ft.dk/api/{}?$inlinecount=allpages&$skip=0".format(ressource)

    while done != True:
        try:
            r = requests.get(url)
            cont = r.json()
            content.append(cont)
            if "odata.nextLink" in cont.keys():
                url = cont['odata.nextLink']
            else:
                done = True
        except Exception as e:
            print(e)

    return content

ressources = ['Afstemning', 'Afstemningstype', 'Aktstykke', 'Aktør', 'AktørAktør', 'AktørAktørRolle',
                'Aktørtype', 'Almdel', 'Dagsordenspunkt', 'DagsordenspunktDokument', 'DagsordenspunktSag',
                'Debat', 'Dokument', 'Fil', 'Forslag', 'KolloneBeskrivelse','Møde', 'MødeAktør', 'Mødestatus', 
                'Mødetype', 'Omtryk', 'Periode', 'Sag', 'SagAktør','SagAktørRolle', 'SagDokument', 
                'SagDokumentRolle', 'Sagskategori', 'Sagsstatus','Sagstrin', 'SagstrinAktør', 'SagstrinAktørRolle', 
                'SagstrinDokument', 'Sagstrinsstatus','Sagstrinstype', 'Sagstype', 'Sambehandlinger', 'Stemme', 'Stemmetype']

for ressource in ressources:
    content = scrape_odaData(ressource)
    with open(r'C:\Users\45287\Desktop\oda_data\{}.json'.format(ressource), 'w') as file:
        json.dump(content, file)