from bs4 import BeautifulSoup
from urllib.request import urlopen as uReq
import base64
from tqdm import tqdm
import requests
import uuid

# 1 = caxias
# 2 = barueri
# 3 = niteroi

i = 0
for i in tqdm(range(3722,5000)):
    i += 1
    #Caxias
    # with uReq("https://nfse.caxias.rs.gov.br/services/parametros/public/captcha") as f:
    #     html = f.read().decode('utf-8')
    # img_data = bytes(html,'utf-8')
    # with open("Data/captcha_groups/1/captcha{}.png".format(i), "wb") as fh:
    #     fh.write(base64.decodebytes(img_data))
    #
    # #Barueri
    # image_url = "https://www.barueri.sp.gov.br/PMB/PortalServicos/WF/imgSeguranca.aspx?1234654="
    # img_data = requests.get(image_url).content
    # print(img_data)
    # with open('Data/captcha_groups/2/captcha{}.png'.format(i), 'wb') as handler:
    #     handler.write(img_data)

    #Niteroi
    # make a random UUID
    guid = str(uuid.uuid4())
    url = "https://nfse.niteroi.rj.gov.br/documentos/CaptchaImage.aspx?guid={}&s=1".format(guid)
    cookies = {'ASP.NET_SessionId': "zznvxdvipivarg41xelimw4u"}

    url1 = "https://nfse.niteroi.rj.gov.br/documentos/verificacao.aspx?mobile=0"
    response1 = requests.get(url1, stream=True, cookies=cookies)
    response = requests.get(url, stream=True, cookies=cookies)

    with open('Data/captcha_groups/3/captcha{}.png'.format(i), 'wb') as handler:
        handler.write(response.content)



