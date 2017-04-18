rsync -av -r -e "ssh -p 22222 -i ~/.ssh/your_private_key"  --exclude secrets.py --exclude *.pyc Your_directory_to/videos3 ml@your_server_ip:/home/your_server_user
