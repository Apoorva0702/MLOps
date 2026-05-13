import requests
import os

username = os.environ.get('JENKINS_USER', 'admin')
password = os.environ.get('JENKINS_PASS')

if not password:
    print("Error: JENKINS_PASS environment variable not set.")
    exit(1)

auth = (username, password)
crumb_url = "http://localhost:9090/crumbIssuer/api/json"
crumb_resp = requests.get(crumb_url, auth=auth, timeout=5)
print("Crumb Resp:", crumb_resp.status_code, crumb_resp.text)
crumb_data = crumb_resp.json()
headers = {crumb_data['crumbRequestField']: crumb_data['crumb']}

webhook_url = "http://localhost:9090/job/fake_news_Retraining/build"
resp = requests.post(webhook_url, auth=auth, headers=headers, timeout=5)
print("Webhook Resp:", resp.status_code, resp.text)
