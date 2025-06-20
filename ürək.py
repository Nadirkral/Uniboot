sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys <PUBKEY>

sudo mkdir -m 0755 -p /etc/apt/keyrings/ 

wget -O- https://example.com/EXAMPLE.gpg |
    gpg --dearmor |
    sudo tee /etc/apt/keyrings/EXAMPLE.gpg > /dev/null
    sudo chmod 644 /etc/apt/keyrings/EXAMPLE.gpg

echo "deb [signed-by=/etc/apt/keyrings/EXAMPLE.gpg] https://example.com/apt stable main" |
    sudo tee /etc/apt/sources.list.d/EXAMPLE.list
    sudo chmod 644 /etc/apt/sources.list.d/EXAMPLE.list

# Optional (you can find the email address / ID using 'apt-key list')
sudo apt-key del support@example.com


sudo gpg --keyserver pgpkeys.mit.edu --recv-key  <PUBKEY>
sudo gpg -a --export <PUBKEY> | sudo apt-key add -
sudo apt-get update


https://github.com/dotnet/runtime