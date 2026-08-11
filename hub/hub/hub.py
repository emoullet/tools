#!/usr/bin/env python3
"""
Arduino Bridge Node (CSV)
==========================

Pont bidirectionnel entre Arduino (USB / CSV) et ROS 2.

Arduino → ROS :
- bouton (test)

ROS → Arduino :
- 
"""

import rclpy
from rclpy.node import Node

from std_msgs.msg import String

import serial
import threading
import time
import geometry_msgs.msg
from std_msgs.msg import Bool, Int8MultiArray, Float32MultiArray, Float32, String, Float32MultiArray

MAX_SERIAL_ARGS = 10

class ArduinoBridgeNode(Node):

    def __init__(self):
        super().__init__('arduino_bridge_node')
        self.loop_rate = self.create_rate(10, self.get_clock())
        
        # ---------------- PUBLISHERS ----------------
        self.digital_input_pub = self.create_publisher(Float32MultiArray, '/hub/digital_input', 10)
        self.analogic_input_pub = self.create_publisher(Float32MultiArray, '/hub/analogic_input', 10)
        

        # ---------------- SUBSCRIBERS ----------------
        self.create_subscription(Float32MultiArray, '/hub/digital_output', self.callback_digital_output, 10)
        self.create_subscription(Float32MultiArray, '/hub/pwm_output', self.callback_pwm_output, 10)
        self.create_subscription(Float32MultiArray, '/hub/setting', self.callback_setting, 10)
        self.time0 = self.get_clock().now().nanoseconds / 1e9
        
        # ---------------- SERIAL ----------------
        try :
            self.serial_port = serial.Serial(
                port='/dev/ttyACM0',
                baudrate=115200,
                timeout=0.1
            )
            self.get_logger().info("Arduino Mega connecté sur /dev/ttyACM0")
        except Exception as e:
            self.get_logger().error(f"Impossible d'ouvrir le port série: {e}")
            raise

        # Thread lecture série
        self.serial_thread = threading.Thread(
            target=self.read_serial_loop,
            daemon=True
        )
        self.serial_thread.start()

    # =====================================================
    #            SERIAL → ROS (LECTURE)
    # =====================================================
    def read_serial_loop(self):
        """
        Lecture des lignes envoyées par l'Arduino (CSV)
        Formats attendus :
        - Bouton: PB,statut
        """
        buffer = ""

        while rclpy.ok():
            try:
                n = self.serial_port.in_waiting
                if n > 0:
                    string = self.serial_port.read(n).decode('utf-8', errors='ignore')
                    buffer += string

                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if line:
                            #print(f"DEBUG RAW LINE: [{line}]")
                            self.handle_csv_line(line)
                else:
                    time.sleep(0.002)
            except Exception as e:
                self.get_logger().error(f"Erreur série: {e}")
                #time.sleep(0.1)

    # =====================================================
    #               CSV DISPATCHER
    # =====================================================
    def handle_csv_line(self, line: str):
        """
        Dispatch selon le premier caractère de la ligne CSV
        """
        if line.startswith("d "):
            self.digitalInput(line)
        elif line.startswith("a "):
            self.analogicInput(line)
        else:
            self.get_logger().warn(f"Ligne CSV inconnue: {line}")

    # =====================================================
    #             ARDUINO → ROS
    # =====================================================
    def digitalInput(self, line: str):
        """
        Format attendu : d <pin_number> <value>
        """
        try:
            msg = Float32MultiArray()
            parts = line.split(' ')
            self.get_logger().info(f"Received digital input: {parts}")  
            size = len(parts)
            for i in range (1, size) :
                msg.data.append(float(parts[i]))
            self.digital_input_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Erreur parsing button: {e}")

    def analogicInput(self, line: str):
        """
        Format attendu : a <pin_number> <value>
        """
        try:
            msg = Float32MultiArray()
            parts = line.split(' ')
            size = len(parts)
            for i in range (1, size) :
                msg.data.append(float(parts[i]))    
            self.analogic_input_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Erreur parsing button: {e}")

    # =====================================================
    #               ROS → ARDUINO
    # =====================================================

    def write_serial_command(self, command: str, values, label: str, require_pairs: bool = False):
        if len(values) > MAX_SERIAL_ARGS:
            self.get_logger().error(
                f"{label} command rejected: {len(values)} arguments exceeds {MAX_SERIAL_ARGS}"
            )
            return

        if require_pairs and len(values) % 2 != 0:
            self.get_logger().error(
                f"{label} command rejected: expected pin/value pairs, got {len(values)} arguments"
            )
            return

        csv_str = command
        for value in values:
            csv_str += f" {int(value)}"
        csv_str += "\n"

        self.serial_port.write(csv_str.encode('utf-8'))
        self.get_logger().info(f"Send {label} command : {csv_str.strip()}")
    
    def callback_digital_output(self, msg: Float32MultiArray):
        """
        Reçoit la commande du topic et l’envoie à l’Arduino.
        """
        #self.get_logger().info(msg.data)
        #self.time0 = self.get_clock().now()
        try:
            """
            Format attendu : D <pin_number> <value>
            """
            self.write_serial_command("D", msg.data, "digital output", require_pairs=True)

        except Exception as e:
            self.get_logger().error(f"Erreur digital output : {e}")
        
        #end_time =  self.get_clock().now() - self.time0
        #self.get_logger().info(f"Time: {end_time}")

    def callback_pwm_output(self, msg: Float32MultiArray):
        """
        Reçoit la commande du topic et l’envoie à l’Arduino.
        """
        #self.get_logger().info(msg.data)
        #self.time0 = self.get_clock().now()
        try:
            """
            Format attendu : P <pin_number> <value>
            """
            self.write_serial_command("P", msg.data, "pwm output", require_pairs=True)

        except Exception as e:
            self.get_logger().error(f"Erreur pwm output : {e}")
        
        #end_time =  self.get_clock().now() - self.time0
        #self.get_logger().info(f"Time: {end_time}")

    def callback_setting(self, msg: Float32MultiArray):
        """
        Reçoit la commande du topic et l’envoie à l’Arduino.
        """
        #self.get_logger().info(msg.data)
        #self.time0 = self.get_clock().now()
        try:
            """
            Format attendu : S <pin_number> <type I/O> (<value>)
            """
            self.write_serial_command("S", msg.data, "settings")

        except Exception as e:
            self.get_logger().error(f"Erreur settings : {e}")
        
        #end_time =  self.get_clock().now() - self.time0
        #self.get_logger().info(f"Time: {end_time}")
    # =====================================================
    #                     CLEANUP
    # =====================================================
    def destroy_node(self):
        if self.serial_port.is_open:
            self.serial_port.close()
        super().destroy_node()

def main(args=None):
    # Init ROS
    rclpy.init(args=args)

    node = ArduinoBridgeNode()
    prev_time = node.get_clock().now()
    try:
        while rclpy.ok():
            rclpy.spin(node)
            rate.sleep()
            time = get_clock().now()
            freq = 1.0 / (time - prev_time).nanoseconds * 1e9
            node.get_logger().info(f"Loop frequency: {freq:.2f} Hz")
            prev_time = get_clock().now()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
