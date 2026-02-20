// MESSAGE VISION_TARGET support class

#pragma once

namespace mavlink {
namespace common {
namespace msg {

/**
 * @brief VISION_TARGET message
 *
 * custom msg - for aruco-marker tracking and landing
 */
struct VISION_TARGET : mavlink::Message {
    static constexpr msgid_t MSG_ID = 150;
    static constexpr size_t LENGTH = 33;
    static constexpr size_t MIN_LENGTH = 33;
    static constexpr uint8_t CRC_EXTRA = 33;
    static constexpr auto NAME = "VISION_TARGET";


    uint64_t timestamp; /*<  timestamp */
    float XY_pgain; /*<  XY axis p gain for marker tracking */
    float XY_igain; /*<  XY axis i gain for marker tracking */
    float XY_dgain; /*<  XY axis d gain for marker tracking */
    float Z_pgain; /*<  Z axis p gain for marker tracking */
    float Z_igain; /*<  Z axis i gain for marker tracking */
    float Z_dgain; /*<  Z axis d gain for marker tracking */
    uint8_t vision_tracking_trig; /*<  vision tracking trig for marker tracking */


    inline std::string get_name(void) const override
    {
            return NAME;
    }

    inline Info get_message_info(void) const override
    {
            return { MSG_ID, LENGTH, MIN_LENGTH, CRC_EXTRA };
    }

    inline std::string to_yaml(void) const override
    {
        std::stringstream ss;

        ss << NAME << ":" << std::endl;
        ss << "  timestamp: " << timestamp << std::endl;
        ss << "  XY_pgain: " << XY_pgain << std::endl;
        ss << "  XY_igain: " << XY_igain << std::endl;
        ss << "  XY_dgain: " << XY_dgain << std::endl;
        ss << "  Z_pgain: " << Z_pgain << std::endl;
        ss << "  Z_igain: " << Z_igain << std::endl;
        ss << "  Z_dgain: " << Z_dgain << std::endl;
        ss << "  vision_tracking_trig: " << +vision_tracking_trig << std::endl;

        return ss.str();
    }

    inline void serialize(mavlink::MsgMap &map) const override
    {
        map.reset(MSG_ID, LENGTH);

        map << timestamp;                     // offset: 0
        map << XY_pgain;                      // offset: 8
        map << XY_igain;                      // offset: 12
        map << XY_dgain;                      // offset: 16
        map << Z_pgain;                       // offset: 20
        map << Z_igain;                       // offset: 24
        map << Z_dgain;                       // offset: 28
        map << vision_tracking_trig;          // offset: 32
    }

    inline void deserialize(mavlink::MsgMap &map) override
    {
        map >> timestamp;                     // offset: 0
        map >> XY_pgain;                      // offset: 8
        map >> XY_igain;                      // offset: 12
        map >> XY_dgain;                      // offset: 16
        map >> Z_pgain;                       // offset: 20
        map >> Z_igain;                       // offset: 24
        map >> Z_dgain;                       // offset: 28
        map >> vision_tracking_trig;          // offset: 32
    }
};

} // namespace msg
} // namespace common
} // namespace mavlink
