import ipaddress
from vimba import *

cam = None
with Vimba.get_instance() as vimba:
    cams = vimba.get_all_cameras()
    for cam in cams:
        with cam:
            ip_feature = cam.get_feature_by_name('GevCurrentIPAddress')
            print(ip_feature)
            ip_int = ip_feature.get()
            print(ip_int)
            ip_str = str(ipaddress.IPv4Address(ip_int))
            print(ip_str)
            if ip_str == "172.16.5.10":
                print('yes')
            else:
                print('no')
