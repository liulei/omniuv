mkdir ~/.pip
cat <<EOF > ~/.pip/pip.conf

 [global]
 trusted-host =  mirrors.aliyun.com
 index-url = http://mirrors.aliyun.com/pypi/simple
EOF