import math
import pandas as pd
import pysftp
import paramiko
import os
import re
import socket
import logging
import traceback
# Mailer
import sendgrid
import os
from sendgrid.helpers.mail import Mail, Email, To, Content

class SaveData:
    def extract_id_from_href(url, include_text):
        '''
        Removes category details in a series/episode URL. 
        e.g. "/title/tt12878838/ -> tt12878838 (include_text=True)
        e.g. "/title/tt12878838/ -> 12878838 (include_text=False)
        '''
        result = re.search(r'\/.*\/(.*)\/?', url)
        if include_text:
            return result.group(1)
        else:
            return result.group(1)[2:]
            
    def trim_href(url):
        '''
        Removes extraneous characters after the ? in a series/episode URL. 
        e.g. "/title/tt12878838/?ref_=nm_flmg_prd_16" -> /title/tt12878838/
        '''
        result = re.search(r'(.*)\?', url)
        
        return result.group(1) if result else url

    def save_persons_to_csv(csv_file, person_list, LOG, include_header=True, mode="a"):
        try:  
            header = ['person_id', 'person_href', 'person_name']
            df = pd.DataFrame(person_list, columns=header)
            # save to csv
            df.to_csv(csv_file, index=False, header=include_header, mode=mode)
        except:
            MyLogger.insert(LOG, f"Failed to save to fullcredits.csv \n {traceback.format_exc()}", logging.ERROR)

    def save_webpage(filename, html, directory, LOG):
        try:
            # save the html file that the driver has open (to use for scraping later)
            # if a folder with the name of the query doesn't exist, create it, then save the file
            if not os.path.isdir(directory):
                os.mkdir(directory)
            with open(f"{directory}/{filename}.html", 'w', encoding='utf-8') as output:
                output.write(html)
        except: 
            MyLogger.insert(LOG, f"Failed to save webpage \n {traceback.format_exc()}", logging.WARNING)

    def save_person_episodes_to_sql(csv_file, person_episodes_list, engine, PROGRESS_LOG, FAIL_LOG):
        try:
            header = ["person_id", "person_href", "person_name", "series_id",
                    "series_href", "series_name", "episode_href", "episode_name", "role", "category", "year"]

            df = pd.DataFrame(person_episodes_list, columns=header)
            # save to csv
            if csv_file:
                df.to_csv(csv_file, mode='a', index=False)
                MyLogger.insert(
                PROGRESS_LOG, "Saved to person_by_episodes.csv", logging.INFO)
            # save to sql database
            df.to_sql("episodes_by_person", engine, if_exists="append",
                    chunksize=1000, index=False, schema="Jenny_Thesis")

        except:
            MyLogger.insert(
                FAIL_LOG, f"Failed to save to person_by_episodes.csv \t Person: {person_episodes_list[0]} \n {traceback.format_exc()}", logging.WARNING)

    def save_shows_to_csv(csv_file, person_shows, LOG, include_header=True, mode="a"):
        '''
        We save the shows to a csv for temporary holding in case of error
        '''
        try:
            # with open(csv_file, "a") as outfile:
            header = ["series_id", "series_href", "series_name"]

            df = pd.DataFrame(person_shows, columns=header)
            # save to csv
            df.to_csv(csv_file, mode=mode, header=include_header, index=False)
            # MyLogger.insert(LOG, "Saved to series.csv", logging.INFO)
        except:
            MyLogger.insert(LOG, "Failed to save to series.csv", logging.WARNING)

class Mailer:
    def send_email(subject, message):
        sg = sendgrid.SendGridAPIClient(api_key="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")
        from_email = Email("jw10@wellesley.edu")  # Change to your verified sender
        to_email = To("jw10@wellesley.edu")  # Change to your recipient
        subject = subject
        content = Content("text/plain", message)
        mail = Mail(from_email, to_email, subject, content)

        # Get a JSON-ready representation of the Mail object
        mail_json = mail.get()

        # Send an HTTP POST request to /mail/send
        response = sg.client.mail.send.post(request_body=mail_json)
        print("Status Code", response.status_code)

class MyLogger:
     def insert(file, text, level):
        '''
        https://stackoverflow.com/questions/49580313/create-a-log-file
        '''
        format = logging.Formatter(
                    '%(asctime)s %(levelname)s %(message)s')
        infoLog = logging.FileHandler(file)
        infoLog.setFormatter(format)
        logger = logging.getLogger(file)
        logger.setLevel(level)

        if not logger.handlers:
            logger.addHandler(infoLog)
            if (level == logging.INFO):
                logger.info(text)
            if (level == logging.ERROR):
                logger.error(text)
            if (level == logging.WARNING):
                logger.warning(text)

        infoLog.close()
        logger.removeHandler(infoLog)

        return 

class EniNode:
    '''
    A helper for communicating within Eni's lab computers, the "nodes"
    '''

    def get_unique_server_id():
        '''
        Server ID is specified by the IP address of Eni's lab computers with IP fixed address 149.130.13.141 until 149.130.13.153. 
        Computer  149.130.13.141 -returns-> 1
        Computer  149.130.13.142 -returns-> 2
        ...
        Computer  149.130.13.152 -returns-> 12
        Computer  149.130.13.153 -returns-> 13
        '''
        ip = socket.gethostbyname(socket.gethostname())
        identifiers = ip.split('.')
        
        return int(identifiers[3])-140

