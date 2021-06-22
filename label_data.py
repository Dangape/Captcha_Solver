from urllib.request import urlopen as uReq
from urllib.request import Request
from urllib import request, parse
import base64
import json
import re
import os
from tqdm import tqdm
import time

for j in range(2,5):

    for i in tqdm(range(4439,5001)):
        try:
            with open("Data/captcha_groups/"+str(j)+"/captcha{}.png".format(i), "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
            base64_pageHtml = str(encoded_string)[2:]

            # Data dict
            data = {
                "userid":"dangape",
                "apikey":"7zFPjaJWmimDTnlEcXjM",
                "data": base64_pageHtml
            }

            # print(data)
            # Dict to Json
            data = json.dumps(data)

            # Convert to String
            data = str(data)

            # Convert string to byte
            data = data.encode('utf-8')

            # Post Method is invoked if data != None
            req =  request.Request('https://api.apitruecaptcha.org/one/gettext', data=data)


            # Response
            resp = request.urlopen(req)
            resp = str(resp.read())[2:-1].split(":")[1].split(",")[0]
            start = '"'
            end = '"'
            resp = resp.split(start)[1].split(end)[0]

            old_file = os.path.join("Data/captcha_groups/"+str(j), "captcha{}.png".format(i))
            new_file = os.path.join("Data/captcha_groups/"+str(j), "{}.png".format(resp))
            if os.path.isfile("Data/captcha_groups/"+str(j)+"/{}.png".format(resp)):
                continue
            else:
                os.rename(old_file, new_file)

        except:
            print("Erro, tentando noavamente em 30s...")
            time.sleep(30)
            with open("Data/captcha_groups/"+str(j)+"/captcha{}.png".format(i), "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
            base64_pageHtml = str(encoded_string)[2:]

            # Data dict
            data = {
                "userid": "dangape",
                "apikey": "7zFPjaJWmimDTnlEcXjM",
                "data": base64_pageHtml
            }

            # print(data)
            # Dict to Json
            data = json.dumps(data)

            # Convert to String
            data = str(data)

            # Convert string to byte
            data = data.encode('utf-8')

            # Post Method is invoked if data != None
            req = request.Request('https://api.apitruecaptcha.org/one/gettext', data=data)

            # Response
            resp = request.urlopen(req)
            resp = str(resp.read())[2:-1].split(":")[1].split(",")[0]
            start = '"'
            end = '"'
            resp = resp.split(start)[1].split(end)[0]

            old_file = os.path.join("Data/captcha_groups/"+str(j), "captcha{}.png".format(i))
            new_file = os.path.join("Data/captcha_groups/"+str(j), "{}.png".format(resp))
            if os.path.isfile("Data/captcha_groups/" + str(j) + "/{}.png".format(resp)):
                continue
            else:
                os.rename(old_file, new_file)
