#pragma once
// MESSAGE VISION_TARGET PACKING

#define MAVLINK_MSG_ID_VISION_TARGET 150


typedef struct __mavlink_vision_target_t {
 uint64_t timestamp; /*<  timestamp*/
 float XY_pgain; /*<  XY axis p gain for marker tracking*/
 float XY_igain; /*<  XY axis i gain for marker tracking*/
 float XY_dgain; /*<  XY axis d gain for marker tracking*/
 float Z_pgain; /*<  Z axis p gain for marker tracking*/
 float Z_igain; /*<  Z axis i gain for marker tracking*/
 float Z_dgain; /*<  Z axis d gain for marker tracking*/
 uint8_t vision_tracking_trig; /*<  vision tracking trig for marker tracking*/
} mavlink_vision_target_t;

#define MAVLINK_MSG_ID_VISION_TARGET_LEN 33
#define MAVLINK_MSG_ID_VISION_TARGET_MIN_LEN 33
#define MAVLINK_MSG_ID_150_LEN 33
#define MAVLINK_MSG_ID_150_MIN_LEN 33

#define MAVLINK_MSG_ID_VISION_TARGET_CRC 33
#define MAVLINK_MSG_ID_150_CRC 33



#if MAVLINK_COMMAND_24BIT
#define MAVLINK_MESSAGE_INFO_VISION_TARGET { \
    150, \
    "VISION_TARGET", \
    8, \
    {  { "timestamp", NULL, MAVLINK_TYPE_UINT64_T, 0, 0, offsetof(mavlink_vision_target_t, timestamp) }, \
         { "XY_pgain", NULL, MAVLINK_TYPE_FLOAT, 0, 8, offsetof(mavlink_vision_target_t, XY_pgain) }, \
         { "XY_igain", NULL, MAVLINK_TYPE_FLOAT, 0, 12, offsetof(mavlink_vision_target_t, XY_igain) }, \
         { "XY_dgain", NULL, MAVLINK_TYPE_FLOAT, 0, 16, offsetof(mavlink_vision_target_t, XY_dgain) }, \
         { "Z_pgain", NULL, MAVLINK_TYPE_FLOAT, 0, 20, offsetof(mavlink_vision_target_t, Z_pgain) }, \
         { "Z_igain", NULL, MAVLINK_TYPE_FLOAT, 0, 24, offsetof(mavlink_vision_target_t, Z_igain) }, \
         { "Z_dgain", NULL, MAVLINK_TYPE_FLOAT, 0, 28, offsetof(mavlink_vision_target_t, Z_dgain) }, \
         { "vision_tracking_trig", NULL, MAVLINK_TYPE_UINT8_T, 0, 32, offsetof(mavlink_vision_target_t, vision_tracking_trig) }, \
         } \
}
#else
#define MAVLINK_MESSAGE_INFO_VISION_TARGET { \
    "VISION_TARGET", \
    8, \
    {  { "timestamp", NULL, MAVLINK_TYPE_UINT64_T, 0, 0, offsetof(mavlink_vision_target_t, timestamp) }, \
         { "XY_pgain", NULL, MAVLINK_TYPE_FLOAT, 0, 8, offsetof(mavlink_vision_target_t, XY_pgain) }, \
         { "XY_igain", NULL, MAVLINK_TYPE_FLOAT, 0, 12, offsetof(mavlink_vision_target_t, XY_igain) }, \
         { "XY_dgain", NULL, MAVLINK_TYPE_FLOAT, 0, 16, offsetof(mavlink_vision_target_t, XY_dgain) }, \
         { "Z_pgain", NULL, MAVLINK_TYPE_FLOAT, 0, 20, offsetof(mavlink_vision_target_t, Z_pgain) }, \
         { "Z_igain", NULL, MAVLINK_TYPE_FLOAT, 0, 24, offsetof(mavlink_vision_target_t, Z_igain) }, \
         { "Z_dgain", NULL, MAVLINK_TYPE_FLOAT, 0, 28, offsetof(mavlink_vision_target_t, Z_dgain) }, \
         { "vision_tracking_trig", NULL, MAVLINK_TYPE_UINT8_T, 0, 32, offsetof(mavlink_vision_target_t, vision_tracking_trig) }, \
         } \
}
#endif

/**
 * @brief Pack a vision_target message
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 *
 * @param timestamp  timestamp
 * @param XY_pgain  XY axis p gain for marker tracking
 * @param XY_igain  XY axis i gain for marker tracking
 * @param XY_dgain  XY axis d gain for marker tracking
 * @param Z_pgain  Z axis p gain for marker tracking
 * @param Z_igain  Z axis i gain for marker tracking
 * @param Z_dgain  Z axis d gain for marker tracking
 * @param vision_tracking_trig  vision tracking trig for marker tracking
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_vision_target_pack(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg,
                               uint64_t timestamp, float XY_pgain, float XY_igain, float XY_dgain, float Z_pgain, float Z_igain, float Z_dgain, uint8_t vision_tracking_trig)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_VISION_TARGET_LEN];
    _mav_put_uint64_t(buf, 0, timestamp);
    _mav_put_float(buf, 8, XY_pgain);
    _mav_put_float(buf, 12, XY_igain);
    _mav_put_float(buf, 16, XY_dgain);
    _mav_put_float(buf, 20, Z_pgain);
    _mav_put_float(buf, 24, Z_igain);
    _mav_put_float(buf, 28, Z_dgain);
    _mav_put_uint8_t(buf, 32, vision_tracking_trig);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_VISION_TARGET_LEN);
#else
    mavlink_vision_target_t packet;
    packet.timestamp = timestamp;
    packet.XY_pgain = XY_pgain;
    packet.XY_igain = XY_igain;
    packet.XY_dgain = XY_dgain;
    packet.Z_pgain = Z_pgain;
    packet.Z_igain = Z_igain;
    packet.Z_dgain = Z_dgain;
    packet.vision_tracking_trig = vision_tracking_trig;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_VISION_TARGET_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_VISION_TARGET;
    return mavlink_finalize_message(msg, system_id, component_id, MAVLINK_MSG_ID_VISION_TARGET_MIN_LEN, MAVLINK_MSG_ID_VISION_TARGET_LEN, MAVLINK_MSG_ID_VISION_TARGET_CRC);
}

/**
 * @brief Pack a vision_target message on a channel
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param timestamp  timestamp
 * @param XY_pgain  XY axis p gain for marker tracking
 * @param XY_igain  XY axis i gain for marker tracking
 * @param XY_dgain  XY axis d gain for marker tracking
 * @param Z_pgain  Z axis p gain for marker tracking
 * @param Z_igain  Z axis i gain for marker tracking
 * @param Z_dgain  Z axis d gain for marker tracking
 * @param vision_tracking_trig  vision tracking trig for marker tracking
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_vision_target_pack_chan(uint8_t system_id, uint8_t component_id, uint8_t chan,
                               mavlink_message_t* msg,
                                   uint64_t timestamp,float XY_pgain,float XY_igain,float XY_dgain,float Z_pgain,float Z_igain,float Z_dgain,uint8_t vision_tracking_trig)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_VISION_TARGET_LEN];
    _mav_put_uint64_t(buf, 0, timestamp);
    _mav_put_float(buf, 8, XY_pgain);
    _mav_put_float(buf, 12, XY_igain);
    _mav_put_float(buf, 16, XY_dgain);
    _mav_put_float(buf, 20, Z_pgain);
    _mav_put_float(buf, 24, Z_igain);
    _mav_put_float(buf, 28, Z_dgain);
    _mav_put_uint8_t(buf, 32, vision_tracking_trig);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_VISION_TARGET_LEN);
#else
    mavlink_vision_target_t packet;
    packet.timestamp = timestamp;
    packet.XY_pgain = XY_pgain;
    packet.XY_igain = XY_igain;
    packet.XY_dgain = XY_dgain;
    packet.Z_pgain = Z_pgain;
    packet.Z_igain = Z_igain;
    packet.Z_dgain = Z_dgain;
    packet.vision_tracking_trig = vision_tracking_trig;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_VISION_TARGET_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_VISION_TARGET;
    return mavlink_finalize_message_chan(msg, system_id, component_id, chan, MAVLINK_MSG_ID_VISION_TARGET_MIN_LEN, MAVLINK_MSG_ID_VISION_TARGET_LEN, MAVLINK_MSG_ID_VISION_TARGET_CRC);
}

/**
 * @brief Encode a vision_target struct
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 * @param vision_target C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_vision_target_encode(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg, const mavlink_vision_target_t* vision_target)
{
    return mavlink_msg_vision_target_pack(system_id, component_id, msg, vision_target->timestamp, vision_target->XY_pgain, vision_target->XY_igain, vision_target->XY_dgain, vision_target->Z_pgain, vision_target->Z_igain, vision_target->Z_dgain, vision_target->vision_tracking_trig);
}

/**
 * @brief Encode a vision_target struct on a channel
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param vision_target C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_vision_target_encode_chan(uint8_t system_id, uint8_t component_id, uint8_t chan, mavlink_message_t* msg, const mavlink_vision_target_t* vision_target)
{
    return mavlink_msg_vision_target_pack_chan(system_id, component_id, chan, msg, vision_target->timestamp, vision_target->XY_pgain, vision_target->XY_igain, vision_target->XY_dgain, vision_target->Z_pgain, vision_target->Z_igain, vision_target->Z_dgain, vision_target->vision_tracking_trig);
}

/**
 * @brief Send a vision_target message
 * @param chan MAVLink channel to send the message
 *
 * @param timestamp  timestamp
 * @param XY_pgain  XY axis p gain for marker tracking
 * @param XY_igain  XY axis i gain for marker tracking
 * @param XY_dgain  XY axis d gain for marker tracking
 * @param Z_pgain  Z axis p gain for marker tracking
 * @param Z_igain  Z axis i gain for marker tracking
 * @param Z_dgain  Z axis d gain for marker tracking
 * @param vision_tracking_trig  vision tracking trig for marker tracking
 */
#ifdef MAVLINK_USE_CONVENIENCE_FUNCTIONS

static inline void mavlink_msg_vision_target_send(mavlink_channel_t chan, uint64_t timestamp, float XY_pgain, float XY_igain, float XY_dgain, float Z_pgain, float Z_igain, float Z_dgain, uint8_t vision_tracking_trig)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_VISION_TARGET_LEN];
    _mav_put_uint64_t(buf, 0, timestamp);
    _mav_put_float(buf, 8, XY_pgain);
    _mav_put_float(buf, 12, XY_igain);
    _mav_put_float(buf, 16, XY_dgain);
    _mav_put_float(buf, 20, Z_pgain);
    _mav_put_float(buf, 24, Z_igain);
    _mav_put_float(buf, 28, Z_dgain);
    _mav_put_uint8_t(buf, 32, vision_tracking_trig);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_VISION_TARGET, buf, MAVLINK_MSG_ID_VISION_TARGET_MIN_LEN, MAVLINK_MSG_ID_VISION_TARGET_LEN, MAVLINK_MSG_ID_VISION_TARGET_CRC);
#else
    mavlink_vision_target_t packet;
    packet.timestamp = timestamp;
    packet.XY_pgain = XY_pgain;
    packet.XY_igain = XY_igain;
    packet.XY_dgain = XY_dgain;
    packet.Z_pgain = Z_pgain;
    packet.Z_igain = Z_igain;
    packet.Z_dgain = Z_dgain;
    packet.vision_tracking_trig = vision_tracking_trig;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_VISION_TARGET, (const char *)&packet, MAVLINK_MSG_ID_VISION_TARGET_MIN_LEN, MAVLINK_MSG_ID_VISION_TARGET_LEN, MAVLINK_MSG_ID_VISION_TARGET_CRC);
#endif
}

/**
 * @brief Send a vision_target message
 * @param chan MAVLink channel to send the message
 * @param struct The MAVLink struct to serialize
 */
static inline void mavlink_msg_vision_target_send_struct(mavlink_channel_t chan, const mavlink_vision_target_t* vision_target)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    mavlink_msg_vision_target_send(chan, vision_target->timestamp, vision_target->XY_pgain, vision_target->XY_igain, vision_target->XY_dgain, vision_target->Z_pgain, vision_target->Z_igain, vision_target->Z_dgain, vision_target->vision_tracking_trig);
#else
    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_VISION_TARGET, (const char *)vision_target, MAVLINK_MSG_ID_VISION_TARGET_MIN_LEN, MAVLINK_MSG_ID_VISION_TARGET_LEN, MAVLINK_MSG_ID_VISION_TARGET_CRC);
#endif
}

#if MAVLINK_MSG_ID_VISION_TARGET_LEN <= MAVLINK_MAX_PAYLOAD_LEN
/*
  This varient of _send() can be used to save stack space by re-using
  memory from the receive buffer.  The caller provides a
  mavlink_message_t which is the size of a full mavlink message. This
  is usually the receive buffer for the channel, and allows a reply to an
  incoming message with minimum stack space usage.
 */
static inline void mavlink_msg_vision_target_send_buf(mavlink_message_t *msgbuf, mavlink_channel_t chan,  uint64_t timestamp, float XY_pgain, float XY_igain, float XY_dgain, float Z_pgain, float Z_igain, float Z_dgain, uint8_t vision_tracking_trig)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char *buf = (char *)msgbuf;
    _mav_put_uint64_t(buf, 0, timestamp);
    _mav_put_float(buf, 8, XY_pgain);
    _mav_put_float(buf, 12, XY_igain);
    _mav_put_float(buf, 16, XY_dgain);
    _mav_put_float(buf, 20, Z_pgain);
    _mav_put_float(buf, 24, Z_igain);
    _mav_put_float(buf, 28, Z_dgain);
    _mav_put_uint8_t(buf, 32, vision_tracking_trig);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_VISION_TARGET, buf, MAVLINK_MSG_ID_VISION_TARGET_MIN_LEN, MAVLINK_MSG_ID_VISION_TARGET_LEN, MAVLINK_MSG_ID_VISION_TARGET_CRC);
#else
    mavlink_vision_target_t *packet = (mavlink_vision_target_t *)msgbuf;
    packet->timestamp = timestamp;
    packet->XY_pgain = XY_pgain;
    packet->XY_igain = XY_igain;
    packet->XY_dgain = XY_dgain;
    packet->Z_pgain = Z_pgain;
    packet->Z_igain = Z_igain;
    packet->Z_dgain = Z_dgain;
    packet->vision_tracking_trig = vision_tracking_trig;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_VISION_TARGET, (const char *)packet, MAVLINK_MSG_ID_VISION_TARGET_MIN_LEN, MAVLINK_MSG_ID_VISION_TARGET_LEN, MAVLINK_MSG_ID_VISION_TARGET_CRC);
#endif
}
#endif

#endif

// MESSAGE VISION_TARGET UNPACKING


/**
 * @brief Get field timestamp from vision_target message
 *
 * @return  timestamp
 */
static inline uint64_t mavlink_msg_vision_target_get_timestamp(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint64_t(msg,  0);
}

/**
 * @brief Get field XY_pgain from vision_target message
 *
 * @return  XY axis p gain for marker tracking
 */
static inline float mavlink_msg_vision_target_get_XY_pgain(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  8);
}

/**
 * @brief Get field XY_igain from vision_target message
 *
 * @return  XY axis i gain for marker tracking
 */
static inline float mavlink_msg_vision_target_get_XY_igain(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  12);
}

/**
 * @brief Get field XY_dgain from vision_target message
 *
 * @return  XY axis d gain for marker tracking
 */
static inline float mavlink_msg_vision_target_get_XY_dgain(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  16);
}

/**
 * @brief Get field Z_pgain from vision_target message
 *
 * @return  Z axis p gain for marker tracking
 */
static inline float mavlink_msg_vision_target_get_Z_pgain(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  20);
}

/**
 * @brief Get field Z_igain from vision_target message
 *
 * @return  Z axis i gain for marker tracking
 */
static inline float mavlink_msg_vision_target_get_Z_igain(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  24);
}

/**
 * @brief Get field Z_dgain from vision_target message
 *
 * @return  Z axis d gain for marker tracking
 */
static inline float mavlink_msg_vision_target_get_Z_dgain(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  28);
}

/**
 * @brief Get field vision_tracking_trig from vision_target message
 *
 * @return  vision tracking trig for marker tracking
 */
static inline uint8_t mavlink_msg_vision_target_get_vision_tracking_trig(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint8_t(msg,  32);
}

/**
 * @brief Decode a vision_target message into a struct
 *
 * @param msg The message to decode
 * @param vision_target C-struct to decode the message contents into
 */
static inline void mavlink_msg_vision_target_decode(const mavlink_message_t* msg, mavlink_vision_target_t* vision_target)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    vision_target->timestamp = mavlink_msg_vision_target_get_timestamp(msg);
    vision_target->XY_pgain = mavlink_msg_vision_target_get_XY_pgain(msg);
    vision_target->XY_igain = mavlink_msg_vision_target_get_XY_igain(msg);
    vision_target->XY_dgain = mavlink_msg_vision_target_get_XY_dgain(msg);
    vision_target->Z_pgain = mavlink_msg_vision_target_get_Z_pgain(msg);
    vision_target->Z_igain = mavlink_msg_vision_target_get_Z_igain(msg);
    vision_target->Z_dgain = mavlink_msg_vision_target_get_Z_dgain(msg);
    vision_target->vision_tracking_trig = mavlink_msg_vision_target_get_vision_tracking_trig(msg);
#else
        uint8_t len = msg->len < MAVLINK_MSG_ID_VISION_TARGET_LEN? msg->len : MAVLINK_MSG_ID_VISION_TARGET_LEN;
        memset(vision_target, 0, MAVLINK_MSG_ID_VISION_TARGET_LEN);
    memcpy(vision_target, _MAV_PAYLOAD(msg), len);
#endif
}
