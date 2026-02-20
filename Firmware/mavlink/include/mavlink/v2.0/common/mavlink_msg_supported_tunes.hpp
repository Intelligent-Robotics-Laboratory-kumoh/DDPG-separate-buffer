// MESSAGE SUPPORTED_TUNES support class

#pragma once

namespace mavlink {
namespace common {
namespace msg {

/**
 * @brief SUPPORTED_TUNES message
 *
 * Tune formats supported by vehicle. This should be emitted as response to MAV_CMD_REQUEST_MESSAGE.
 */
struct SUPPORTED_TUNES : mavlink::Message {
    static constexpr msgid_t MSG_ID = 401;
    static constexpr size_t LENGTH = 202;
    static constexpr size_t MIN_LENGTH = 202;
    static constexpr uint8_t CRC_EXTRA = 143;
    static constexpr auto NAME = "SUPPORTED_TUNES";


    uint8_t target_system; /*<  System ID */
    uint8_t target_component; /*<  Component ID */
    std::array<uint8_t, 200> format; /*<  Array of supported tune formats */


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
        ss << "  target_system: " << +target_system << std::endl;
        ss << "  target_component: " << +target_component << std::endl;
        ss << "  format: [" << to_string(format) << "]" << std::endl;

        return ss.str();
    }

    inline void serialize(mavlink::MsgMap &map) const override
    {
        map.reset(MSG_ID, LENGTH);

        map << target_system;                 // offset: 0
        map << target_component;              // offset: 1
        map << format;                        // offset: 2
    }

    inline void deserialize(mavlink::MsgMap &map) override
    {
        map >> target_system;                 // offset: 0
        map >> target_component;              // offset: 1
        map >> format;                        // offset: 2
    }
};

} // namespace msg
} // namespace common
} // namespace mavlink
