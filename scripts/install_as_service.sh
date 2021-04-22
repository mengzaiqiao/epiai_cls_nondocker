sudo cp ./epiai_cls.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable epiai_cls
sudo systemctl start epiai_cls