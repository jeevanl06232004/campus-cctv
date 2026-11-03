import os
import json
import time
import threading
import urllib.parse
from pathlib import Path
import cv2
import requests

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.json"
INCIDENTS_DIR = BASE_DIR / "incidents"
INCIDENTS_FILE = INCIDENTS_DIR / "incidents.json"
INCIDENTS_DIR.mkdir(exist_ok=True)

DEFAULT_CONFIG = {
    "master_password": "admin",
    "authorities_count": 1,
    "on_duty_id": 1,
    "authorities": [
        {
            "id": 1,
            "name": "Authority Person #1",
            "phone": "",
            "email": "",
            "is_on_duty": True
        }
    ],
    "cooldown_seconds": 25,
    "sensitivity": "medium",
    "sound_alarm": True,
    "smtp": {
        "enabled": False,
        "host": "smtp.gmail.com",
        "port": 587,
        "sender_email": "",
        "sender_password": ""
    }
}


class AlertManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.config = self._load_config()
        self.last_alert_time = 0.0
        self.last_authority_message = None
        self.recent_dispatches = []
        self.alarm_active = False
        self.alarm_timestamp = 0.0
        self.alarm_incident_id = None
        self._ensure_incidents_file()

    def _load_config(self):
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cfg = DEFAULT_CONFIG.copy()
                    cfg.update(data)

                    raw_auths = data.get("authorities", [])
                    # Support dynamic selection: 1 authority asks for 1, 3 authorities asks for 3!
                    auth_count = int(data.get("authorities_count") or len(raw_auths) or 1)
                    auth_count = max(1, min(10, auth_count))
                    cfg["authorities_count"] = auth_count

                    on_duty_id = int(data.get("on_duty_id", 1))
                    if on_duty_id > auth_count:
                        on_duty_id = 1
                    cfg["on_duty_id"] = on_duty_id

                    existing_by_id = {a.get("id"): a for a in raw_auths if isinstance(a, dict)}
                    clean_auths = []
                    for i in range(1, auth_count + 1):
                        if i in existing_by_id:
                            item = existing_by_id[i]
                            clean_auths.append({
                                "id": i,
                                "name": item.get("name", f"Authority Person #{i}"),
                                "phone": item.get("phone", ""),
                                "email": item.get("email", ""),
                                "is_on_duty": (i == on_duty_id)
                            })
                        else:
                            clean_auths.append({
                                "id": i,
                                "name": f"Authority Person #{i}",
                                "phone": "",
                                "email": "",
                                "is_on_duty": (i == on_duty_id)
                            })
                    cfg["authorities"] = clean_auths
                    return cfg
            except Exception as e:
                print(f"[Notifier] Error reading config: {e}")
        return DEFAULT_CONFIG.copy()

    def save_config(self, new_config=None):
        with self.lock:
            if new_config:
                self.config.update(new_config)
            try:
                with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                    json.dump(self.config, f, indent=4)
                return True, "Configuration saved successfully."
            except Exception as e:
                return False, f"Error saving configuration: {e}"

    def get_on_duty_authority(self):
        """Returns the currently active On-Duty authority person."""
        on_duty_id = self.config.get("on_duty_id", 1)
        for a in self.config.get("authorities", []):
            if a.get("id") == on_duty_id:
                res = a.copy()
                res["is_on_duty"] = True
                return res
        # Fallback to slot 1
        auths = self.config.get("authorities", [])
        if auths:
            res = auths[0].copy()
            res["is_on_duty"] = True
            return res
        return {"id": 1, "name": "Authority Person #1", "phone": "", "email": "", "is_on_duty": True}

    def get_authority_person(self):
        """Backward compatibility for telemetry and status displays."""
        with self.lock:
            auth = self.get_on_duty_authority()
            contact_str = auth.get("phone") or auth.get("email") or "Not configured"
            if auth.get("phone") and auth.get("email"):
                contact_str = f"{auth.get('phone')} | {auth.get('email')}"
            return {
                "id": auth.get("id", 1),
                "name": auth.get("name", "Authority Person #1"),
                "phone": auth.get("phone", ""),
                "email": auth.get("email", ""),
                "contact": contact_str,
                "password_set": True,
                "is_on_duty": True
            }

    def get_authorities(self):
        """Returns the configured authority slots with on-duty status indicated."""
        with self.lock:
            on_duty_id = self.config.get("on_duty_id", 1)
            auth_count = self.config.get("authorities_count", len(self.config.get("authorities", [])))
            sanitized = []
            for a in self.config.get("authorities", []):
                item = a.copy()
                item["is_on_duty"] = (item.get("id") == on_duty_id)
                sanitized.append(item)
            return {
                "authorities_count": auth_count,
                "on_duty_id": on_duty_id,
                "on_duty_authority": self.get_on_duty_authority(),
                "authorities": sanitized
            }

    def save_all_authorities(self, count, authorities_data, on_duty_id=1, password=""):
        """
        Dynamically sets and saves the exact number of authorities (1 to 10).
        If user selects 1, saves only 1 authority.
        If user selects 3, saves exactly 3 authorities.
        """
        count = int(count)
        if count < 1 or count > 10:
            return False, "Number of authorities must be between 1 and 10."

        with self.lock:
            master_pw = self.config.get("master_password", "admin")
            if password != master_pw:
                return False, "❌ Incorrect Master Security Password! Authority data is protected."

            on_duty_id = int(on_duty_id)
            if on_duty_id < 1 or on_duty_id > count:
                on_duty_id = 1

            new_list = []
            for i in range(1, count + 1):
                entry = None
                if isinstance(authorities_data, list) and (i - 1 < len(authorities_data)):
                    entry = authorities_data[i - 1]
                elif isinstance(authorities_data, dict):
                    entry = authorities_data.get(str(i)) or authorities_data.get(i)

                name = (entry.get("name") if entry else "") or f"Authority Person #{i}"
                phone = (entry.get("phone") if entry else "") or ""
                email = (entry.get("email") if entry else "") or ""

                new_list.append({
                    "id": i,
                    "name": name.strip(),
                    "phone": phone.strip(),
                    "email": email.strip(),
                    "is_on_duty": (i == on_duty_id)
                })

            self.config["authorities_count"] = count
            self.config["on_duty_id"] = on_duty_id
            self.config["authorities"] = new_list

            try:
                with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                    json.dump(self.config, f, indent=4)
                return True, f"✅ Successfully configured {count} authority person(s)!"
            except Exception as e:
                return False, f"Error saving config: {e}"

    def update_authority_slot(self, slot_id, name, phone, email, is_on_duty=False, password=""):
        """
        Updates an authority slot (1 to 10).
        Requires single Master Security Password.
        """
        slot_id = int(slot_id)
        if slot_id < 1 or slot_id > 10:
            return False, "Authority ID must be between 1 and 10."

        with self.lock:
            master_pw = self.config.get("master_password", "admin")
            if password != master_pw:
                return False, "❌ Incorrect Master Security Password! Authority data is protected."

            authorities = self.config.get("authorities", DEFAULT_AUTHORITIES)
            found = False
            for a in authorities:
                if a.get("id") == slot_id:
                    a["name"] = name.strip() or f"Authority Person #{slot_id}"
                    a["phone"] = phone.strip()
                    a["email"] = email.strip()
                    found = True
                    break

            if not found:
                authorities.append({
                    "id": slot_id,
                    "name": name.strip() or f"Authority Person #{slot_id}",
                    "phone": phone.strip(),
                    "email": email.strip(),
                    "is_on_duty": False
                })

            if is_on_duty:
                self.config["on_duty_id"] = slot_id

            # Keep is_on_duty synchronized
            for a in authorities:
                a["is_on_duty"] = (a.get("id") == self.config.get("on_duty_id", 1))

            self.config["authorities"] = authorities

            try:
                with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                    json.dump(self.config, f, indent=4)
                return True, f"✅ Authority Person #{slot_id} ('{name}') updated successfully!"
            except Exception as e:
                return False, f"Error writing config: {e}"

    def set_on_duty(self, slot_id, password=""):
        """Quickly designates which of the 10 authorities is currently On-Duty."""
        slot_id = int(slot_id)
        if slot_id < 1 or slot_id > 10:
            return False, "Authority ID must be between 1 and 10."

        with self.lock:
            master_pw = self.config.get("master_password", "admin")
            if password and password != master_pw:
                return False, "❌ Incorrect Master Security Password."

            self.config["on_duty_id"] = slot_id
            for a in self.config.get("authorities", []):
                a["is_on_duty"] = (a.get("id") == slot_id)

            try:
                with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                    json.dump(self.config, f, indent=4)
                auth = self.get_on_duty_authority()
                return True, f"✅ Authority Person #{slot_id} ('{auth.get('name')}') is now ON-DUTY!"
            except Exception as e:
                return False, f"Error saving on-duty authority: {e}"

    def update_authority_person(self, auth_data, provided_password=None):
        """Backward compatibility for single-person update requests."""
        slot_id = int(auth_data.get("id") or self.config.get("on_duty_id", 1))
        name = auth_data.get("name", "")
        phone = auth_data.get("phone", auth_data.get("contact", ""))
        email = auth_data.get("email", "")
        pw = provided_password or auth_data.get("password") or auth_data.get("current_password", "")
        is_on_duty = auth_data.get("is_on_duty", True)
        return self.update_authority_slot(slot_id, name, phone, email, is_on_duty=is_on_duty, password=pw)

    def _ensure_incidents_file(self):
        if not INCIDENTS_FILE.exists():
            with open(INCIDENTS_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)

    def get_incidents(self, limit=50):
        try:
            if INCIDENTS_FILE.exists():
                with open(INCIDENTS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return sorted(data, key=lambda x: x.get("timestamp_epoch", 0), reverse=True)[:limit]
        except Exception as e:
            print(f"[Notifier] Error reading incidents: {e}")
        return []

    def log_incident(self, incident_data):
        try:
            incidents = []
            if INCIDENTS_FILE.exists():
                with open(INCIDENTS_FILE, "r", encoding="utf-8") as f:
                    incidents = json.load(f)
            incidents.append(incident_data)
            if len(incidents) > 200:
                incidents = incidents[-200:]
            with open(INCIDENTS_FILE, "w", encoding="utf-8") as f:
                json.dump(incidents, f, indent=2)
        except Exception as e:
            print(f"[Notifier] Error writing incident: {e}")

    def silence_alarm(self):
        """Silences the active audio/visual notification alarm."""
        with self.lock:
            self.alarm_active = False
            self.alarm_incident_id = None
            return True, "Alarm silenced."

    def get_alarm_status(self):
        """Returns the current real-time notification alarm status."""
        with self.lock:
            now = time.time()
            is_active = self.alarm_active and (now - self.alarm_timestamp < 120)  # active for up to 2 mins unless silenced
            return {
                "alarm_active": is_active,
                "alarm_timestamp": self.alarm_timestamp,
                "incident_id": self.alarm_incident_id,
                "last_message": self.last_authority_message
            }

    def _send_smtp_email_async(self, recipient_email, subject, body_text, snapshot_path=None):
        def _worker():
            try:
                smtp_cfg = self.config.get("smtp", {})
                if not smtp_cfg.get("enabled"):
                    return
                sender_email = smtp_cfg.get("sender_email")
                sender_password = smtp_cfg.get("sender_password")
                smtp_host = smtp_cfg.get("host", "smtp.gmail.com")
                smtp_port = int(smtp_cfg.get("port", 587))

                if not sender_email or not sender_password or not recipient_email:
                    return

                msg = MIMEMultipart()
                msg["From"] = f"Campus CCTV Security <{sender_email}>"
                msg["To"] = recipient_email
                msg["Subject"] = subject

                msg.attach(MIMEText(body_text, "plain"))

                if snapshot_path and Path(snapshot_path).exists():
                    try:
                        with open(snapshot_path, "rb") as f:
                            img_data = f.read()
                            img_part = MIMEImage(img_data, name=Path(snapshot_path).name)
                            img_part.add_header("Content-Disposition", "attachment", filename=Path(snapshot_path).name)
                            msg.attach(img_part)
                    except Exception as img_err:
                        print(f"[Notifier] Email attachment error: {img_err}")

                with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                    server.starttls()
                    server.login(sender_email, sender_password)
                    server.send_message(msg)
                print(f"[Notifier] Real SMTP email successfully delivered to {recipient_email}")
            except Exception as e:
                print(f"[Notifier] SMTP email dispatch notice: {e}")

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _send_sms_async(self, phone, message_text):
        def _worker():
            try:
                clean_phone = "".join(c for c in phone if c.isdigit())
                if not clean_phone:
                    return
                int_phone = clean_phone
                if len(int_phone) == 10:
                    int_phone = "+91" + int_phone
                elif not int_phone.startswith("+"):
                    int_phone = "+" + int_phone

                sms_body = message_text[:155]

                # 1. Textbelt SMS attempt
                try:
                    tb_key = self.config.get("textbelt_api_key", "textbelt")
                    resp = requests.post(
                        "https://textbelt.com/text",
                        data={"phone": int_phone, "message": sms_body, "key": tb_key},
                        timeout=5
                    )
                    res_json = resp.json()
                    if res_json.get("success"):
                        print(f"[Notifier] SMS delivered to {int_phone} via Textbelt: {res_json}")
                    else:
                        print(f"[Notifier] Textbelt SMS note: {res_json.get('error', 'Quotas or network')}")
                except Exception as tb_err:
                    print(f"[Notifier] Textbelt dispatch note: {tb_err}")

                # 2. Fast2SMS (Indian Gateway) if configured
                fast2sms_key = self.config.get("fast2sms_api_key")
                if fast2sms_key and len(clean_phone) >= 10:
                    try:
                        p10 = clean_phone[-10:]
                        f_url = "https://www.fast2sms.com/dev/bulkV2"
                        f_payload = {
                            "variables_values": sms_body[:100],
                            "route": "otp",
                            "numbers": p10
                        }
                        f_headers = {"authorization": fast2sms_key}
                        f_resp = requests.post(f_url, data=f_payload, headers=f_headers, timeout=5)
                        print(f"[Notifier] Fast2SMS status: {f_resp.status_code}")
                    except Exception as f_err:
                        print(f"[Notifier] Fast2SMS note: {f_err}")

                # 3. Twilio SMS if configured
                tw_sid = self.config.get("twilio_sid")
                tw_token = self.config.get("twilio_token")
                tw_from = self.config.get("twilio_from")
                if tw_sid and tw_token and tw_from:
                    try:
                        tw_url = f"https://api.twilio.com/2010-04-01/Accounts/{tw_sid}/Messages.json"
                        tw_resp = requests.post(
                            tw_url,
                            auth=(tw_sid, tw_token),
                            data={"From": tw_from, "To": int_phone, "Body": sms_body},
                            timeout=6
                        )
                        print(f"[Notifier] Twilio SMS status: {tw_resp.status_code}")
                    except Exception as tw_err:
                        print(f"[Notifier] Twilio note: {tw_err}")
            except Exception as e:
                print(f"[Notifier] SMS worker note: {e}")

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def trigger_alert(self, alert_type, details, frame=None, force=False):
        now = time.time()
        cooldown = self.config.get("cooldown_seconds", 25)

        with self.lock:
            if not force and (now - self.last_alert_time < cooldown):
                return False, f"Cooldown active. {int(cooldown - (now - self.last_alert_time))}s remaining."
            self.last_alert_time = now

        incident_id = f"inc_{int(now)}_{alert_type.lower()}"
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

        # Get the currently designated On-Duty authority
        auth = self.get_on_duty_authority()
        auth_name = auth.get("name") or f"Authority Person #{auth.get('id', 1)}"
        auth_phone = auth.get("phone", "")
        auth_email = auth.get("email", "")

        contact_display = auth_phone or auth_email or "Local System Log"
        if auth_phone and auth_email:
            contact_display = f"{auth_phone} | {auth_email}"

        snapshot_filename = f"{incident_id}.jpg"
        snapshot_path = INCIDENTS_DIR / snapshot_filename

        if frame is not None:
            try:
                cv2.imwrite(str(snapshot_path), frame)
            except Exception as e:
                print(f"[Notifier] Error saving frame: {e}")
                snapshot_filename = None
        else:
            snapshot_filename = None

        if alert_type == "FIGHT_DETECTED":
            alert_title = "VIOLENT FIGHT DETECTED"
        elif alert_type in ("ACTION_CHECK", "AGGRESSIVE_ACTION"):
            alert_title = "AGGRESSIVE ACTION DETECTED"
        elif alert_type == "HIGH_MOTION":
            alert_title = "HIGH MOTION ANOMALY"
        else:
            alert_title = "EMERGENCY ACTION ALARM"

        persons_str = ", ".join(details.get("persons", [])) if details.get("persons") else f"{auth_name} Verified Action"

        text_message = (
            f"🚨 CAMPUS CCTV EMERGENCY NOTIFICATION ALARM\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"TO ON-DUTY AUTHORITY: {auth_name.upper()} (Person #{auth.get('id', 1)})\n"
            f"AUTHORIZED PHONE: {auth_phone or 'N/A'}\n"
            f"AUTHORIZED EMAIL: {auth_email or 'N/A'}\n"
            f"INCIDENT STATUS : {alert_title}\n"
            f"CAMERA LOCATION : {details.get('location', 'Camera 01 - Main Quad')}\n"
            f"ACTION TIME     : {timestamp_str}\n"
            f"ACTION TARGETS  : {persons_str}\n"
            f"ACTION INTENSITY: {details.get('score', 0):.1f}%\n"
            f"SECURITY ADVICE : Immediate intervention requested on campus."
        )

        clean_phone = "".join(c for c in auth_phone if c.isdigit())
        if len(clean_phone) == 10 and not clean_phone.startswith("91"):
            clean_phone = "91" + clean_phone

        whatsapp_url = f"https://api.whatsapp.com/send?phone={clean_phone}&text={urllib.parse.quote(text_message)}" if clean_phone else ""
        sms_url = f"sms:+{clean_phone}?body={urllib.parse.quote(text_message)}" if clean_phone else ""

        mailto_subject = urllib.parse.quote(f"🚨 CCTV SECURITY ALARM: {alert_title} - {timestamp_str}")
        mailto_body = urllib.parse.quote(text_message)
        mailto_url = f"mailto:{auth_email}?subject={mailto_subject}&body={mailto_body}" if auth_email else ""

        # Instant Push Notification to phone via ntfy.sh
        topic_name = f"cctv_{clean_phone}" if clean_phone else "cctv_campus_alert_jeevan"
        push_url = f"https://ntfy.sh/{topic_name}"
        try:
            requests.post(
                push_url,
                data=f"🚨 {alert_title}\nDispatched to: {auth_name}\nPhone: {auth_phone}\nEmail: {auth_email}\nTime: {timestamp_str}\nLocation: {details.get('location', 'Camera 01')}".encode("utf-8"),
                headers={
                    "Title": f"CCTV ALARM: {alert_title}",
                    "Priority": "urgent",
                    "Tags": "warning,rotating_light"
                },
                timeout=4
            )
            print(f"[Notifier] Dispatched push notification to {push_url}")
        except Exception as push_err:
            print(f"[Notifier] Push dispatch notice: {push_err}")

        # Real SMS dispatch
        if clean_phone:
            self._send_sms_async(clean_phone, text_message)

        # Real SMTP background email if configured
        if auth_email:
            self._send_smtp_email_async(
                recipient_email=auth_email,
                subject=f"🚨 CCTV EMERGENCY ALARM: {alert_title} ({timestamp_str})",
                body_text=text_message,
                snapshot_path=str(snapshot_path) if snapshot_filename else None
            )

        authority_dispatch_record = {
            "authority_id": auth.get("id", 1),
            "authority_name": auth_name,
            "authority_contact": contact_display,
            "authority_phone": auth_phone,
            "authority_email": auth_email,
            "message": text_message,
            "sms_url": sms_url,
            "whatsapp_url": whatsapp_url,
            "mailto_url": mailto_url,
            "push_url": push_url,
            "timestamp": timestamp_str,
            "type": alert_type,
            "snapshot": snapshot_filename,
            "alarm_active": True
        }

        with self.lock:
            self.last_authority_message = authority_dispatch_record
            self.alarm_active = True
            self.alarm_timestamp = now
            self.alarm_incident_id = incident_id

        incident_record = {
            "id": incident_id,
            "timestamp": timestamp_str,
            "timestamp_epoch": now,
            "type": alert_type,
            "location": details.get("location", "Camera 01 - Main Quad"),
            "source": details.get("source", "Live Webcam"),
            "persons": details.get("persons", []),
            "persons_count": details.get("persons_count", 0),
            "score": details.get("score", 0.0),
            "confidence": details.get("confidence", 0.0),
            "snapshot": snapshot_filename,
            "authority_recipient": {
                "id": auth.get("id", 1),
                "name": auth_name,
                "phone": auth_phone,
                "email": auth_email,
                "contact": contact_display
            },
            "sms_url": sms_url,
            "whatsapp_url": whatsapp_url,
            "mailto_url": mailto_url,
            "dispatched_to": [f"Direct to {auth_name} ({contact_display})", "CCTV Evidence Vault"]
        }

        self.log_incident(incident_record)

        with self.lock:
            self.recent_dispatches.append({
                "time": incident_record["timestamp"],
                "type": incident_record["type"],
                "recipient": auth_name,
                "contact": contact_display,
                "persons": incident_record.get("persons", [])
            })
            if len(self.recent_dispatches) > 20:
                self.recent_dispatches = self.recent_dispatches[-20:]

        return True, incident_id


alert_manager = AlertManager()

