sudo cp ./epiai_cls.service /etc/systemd/system/
sudo service apache2 reload
sudo systemctl daemon-reload
sudo systemctl start epiai_cls