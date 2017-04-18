rsync -av -r -e "ssh -p 22222 -i ~/.ssh/your_private_key"  --exclude secrets.py --exclude *.pyc /Your Directory to Behavioral_Cloning/Behavioral_Cloning ml@your_server_ip:/home/your_server_user
