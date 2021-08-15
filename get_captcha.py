from urllib.request import urlopen as uReq
import base64
from tqdm import tqdm
import requests
import uuid
from bs4 import BeautifulSoup as bs

# 1 = caxias
# 2 = barueri
# 3 = niteroi
# 7 = aparecido de goiania
# 8 = belo horizonte
# 9 = blumenau

i = 0
for i in tqdm(range(247,2000)):
    i += 1
    # # Caxias
    with uReq("https://nfse.caxias.rs.gov.br/services/parametros/public/captcha") as f:
        html = f.read().decode('utf-8')
    img_data = bytes(html,'utf-8')
    with open("Data/captcha_groups/1/captcha{}.png".format(i), "wb") as fh:
        fh.write(base64.decodebytes(img_data))
    #
    # #Barueri
    # image_url = "https://www.barueri.sp.gov.br/PMB/PortalServicos/WF/imgSeguranca.aspx?1234654="
    # img_data = requests.get(image_url).content
    # print(img_data)
    # with open('Data/test_set/caxias/captcha{}.png'.format(i), 'wb') as handler:
    #     handler.write(img_data)

    # #Niteroi
    # # make a random UUID
    # guid = str(uuid.uuid4())
    # url = "https://nfse.niteroi.rj.gov.br/documentos/CaptchaImage.aspx?guid={}&s=1".format(guid)
    # cookies = {'ASP.NET_SessionId': "zznvxdvipivarg41xelimw4u"}
    #
    # url1 = "https://nfse.niteroi.rj.gov.br/documentos/verificacao.aspx?mobile=0"
    # response1 = requests.get(url1, stream=True, cookies=cookies)
    # response = requests.get(url, stream=True, cookies=cookies)
    #
    # with open('Data/captcha_groups/3/captcha{}.png'.format(i), 'wb') as handler:
    #     handler.write(response.content)

    #Aparecida goiania
    soup = bs(requests.get("https://www.issnetonline.com.br/aparecida/online/NotaDigital/VerificaAutenticidade.aspx").content, "html.parser")

    img = soup.find("img", { "id" : "imgImagemConfirma" })
    img_url = "https://www.issnetonline.com.br/aparecida/online/NotaDigital/" + img.attrs.get("src")
    img_data = requests.get(img_url).content
    # print(img_data)

    with open("Data/test_set/7/captcha{}.png".format(i), "wb") as fh:
        fh.write(img_data)

    # # Belo horizonte
    # image_url = "https://bhissdigital.pbh.gov.br/nfse/captcha.jpg?pk="
    # img_data = requests.get(image_url).content
    # with open('Data/not_trained/8/captcha{}.png'.format(i), 'wb') as handler:
    #     handler.write(img_data)

    #Blumenau
    # guid = str(uuid.uuid4())
    # url = "https://nfse.blumenau.sc.gov.br/contrib/app/nfse/captcha?x={}".format(guid)
    # cookies = {'ASP.NET_SessionId': "5l2cokkpnn5clx3l34i4gqp2"}
    #
    # url1 = "https://nfse.blumenau.sc.gov.br/contrib/app/nfse/impressao_externa?s=25170558&e=80680093000181&f=1MRQHCTK"
    # response1 = requests.get(url1, stream=True, cookies=cookies)
    # response = requests.get(url, stream=True, cookies=cookies)
    #
    # with open('Data/not_trained/9/captcha{}.png'.format(i), 'wb') as handler:
    #     handler.write(response.content)


