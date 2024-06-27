#include <dji_control.hpp>
#include <dji_status.hpp>
#include <dji_version.hpp>
#include <unordered_map>
#include <string>

std::unordered_map<std::string, int> constants_map = {
    {"DJISDK::Control::HORIZONTAL_ANGLE", DJI::OSDK::Control::HORIZONTAL_ANGLE},
    {"DJISDK::Control::HORIZONTAL_VELOCITY", DJI::OSDK::Control::HORIZONTAL_VELOCITY},
    {"DJISDK::Control::HORIZONTAL_POSITION", DJI::OSDK::Control::HORIZONTAL_POSITION},
    {"DJISDK::Control::HORIZONTAL_ANGULAR_RATE", DJI::OSDK::Control::HORIZONTAL_ANGULAR_RATE},
    {"DJISDK::Control::VERTICAL_VELOCITY", DJI::OSDK::Control::VERTICAL_VELOCITY},
    {"DJISDK::Control::VERTICAL_POSITION", DJI::OSDK::Control::VERTICAL_POSITION},
    {"DJISDK::Control::VERTICAL_THRUST", DJI::OSDK::Control::VERTICAL_THRUST},
    {"DJISDK::Control::YAW_ANGLE", DJI::OSDK::Control::YAW_ANGLE},
    {"DJISDK::Control::YAW_RATE", DJI::OSDK::Control::YAW_RATE},
    {"DJISDK::Control::HORIZONTAL_GROUND", DJI::OSDK::Control::HORIZONTAL_GROUND},
    {"DJISDK::Control::HORIZONTAL_BODY", DJI::OSDK::Control::HORIZONTAL_BODY},
    {"DJISDK::Control::STABLE_DISABLE", DJI::OSDK::Control::STABLE_DISABLE},
    {"DJISDK::Control::STABLE_ENABLE", DJI::OSDK::Control::STABLE_ENABLE},

    {"DJISDK::DisplayMode::MODE_MANUAL_CTRL", DJI::OSDK::VehicleStatus::DisplayMode::MODE_MANUAL_CTRL},
    {"DJISDK::DisplayMode::MODE_ATTITUDE", DJI::OSDK::VehicleStatus::DisplayMode::MODE_ATTITUDE},
    {"DJISDK::DisplayMode::MODE_P_GPS", DJI::OSDK::VehicleStatus::DisplayMode::MODE_P_GPS},
    {"DJISDK::DisplayMode::MODE_HOTPOINT_MODE", DJI::OSDK::VehicleStatus::DisplayMode::MODE_HOTPOINT_MODE},
    {"DJISDK::DisplayMode::MODE_ASSISTED_TAKEOFF", DJI::OSDK::VehicleStatus::DisplayMode::MODE_ASSISTED_TAKEOFF},
    {"DJISDK::DisplayMode::MODE_AUTO_TAKEOFF", DJI::OSDK::VehicleStatus::DisplayMode::MODE_AUTO_TAKEOFF},
    {"DJISDK::DisplayMode::MODE_AUTO_LANDING", DJI::OSDK::VehicleStatus::DisplayMode::MODE_AUTO_LANDING},
    {"DJISDK::DisplayMode::MODE_NAVI_GO_HOME", DJI::OSDK::VehicleStatus::DisplayMode::MODE_NAVI_GO_HOME},
    {"DJISDK::DisplayMode::MODE_NAVI_SDK_CTRL", DJI::OSDK::VehicleStatus::DisplayMode::MODE_NAVI_SDK_CTRL},
    {"DJISDK::DisplayMode::MODE_FORCE_AUTO_LANDING", DJI::OSDK::VehicleStatus::DisplayMode::MODE_FORCE_AUTO_LANDING},
    {"DJISDK::DisplayMode::MODE_SEARCH_MODE", DJI::OSDK::VehicleStatus::DisplayMode::MODE_SEARCH_MODE},
    {"DJISDK::DisplayMode::MODE_ENGINE_START", DJI::OSDK::VehicleStatus::DisplayMode::MODE_ENGINE_START},

    {"DJISDK::FlightStatus::STATUS_STOPPED", DJI::OSDK::VehicleStatus::FlightStatus::STOPED},
    {"DJISDK::FlightStatus::STATUS_ON_GROUND", DJI::OSDK::VehicleStatus::FlightStatus::ON_GROUND},
    {"DJISDK::FlightStatus::STATUS_IN_AIR", DJI::OSDK::VehicleStatus::FlightStatus::IN_AIR},
    
    {"DJISDK::M100FlightStatus::M100_STATUS_ON_GROUND", DJI::OSDK::VehicleStatus::M100FlightStatus::ON_GROUND_STANDBY},
    {"DJISDK::M100FlightStatus::M100_STATUS_TAKINGOFF", DJI::OSDK::VehicleStatus::M100FlightStatus::TAKEOFF},
    {"DJISDK::M100FlightStatus::M100_STATUS_IN_AIR", DJI::OSDK::VehicleStatus::M100FlightStatus::IN_AIR_STANDBY},
    {"DJISDK::M100FlightStatus::M100_STATUS_LANDING", DJI::OSDK::VehicleStatus::M100FlightStatus::LANDING},
    {"DJISDK::M100FlightStatus::M100_STATUS_FINISHED_LANDING", DJI::OSDK::VehicleStatus::M100FlightStatus::FINISHING_LANDING}
};

extern "C" {
    int get_constant(const char* name) {
        auto it = constants_map.find(name);
        if (it != constants_map.end()) {
            return it->second;
        } else {
            return -1;
        }
    }
}