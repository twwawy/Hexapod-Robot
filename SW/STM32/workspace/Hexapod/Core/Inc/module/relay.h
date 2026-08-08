#ifndef MODULE_RELAY_H
#define MODULE_RELAY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

/**
 * @brief Relay channels connected to PC2, PC3, and PC4.
 *
 * The relay inputs are active-high:
 * GPIO HIGH = relay ON, GPIO LOW = relay OFF.
 */
typedef enum
{
    RELAY_INA = 0,
    RELAY_INB,
    RELAY_INC,
    RELAY_CHANNEL_COUNT
} Relay_Channel_t;

/**
 * @brief Turn every relay OFF.
 * @note Call once after MX_GPIO_Init().
 */
void Relay_Init(void);

/** @brief Set one relay to the requested state. */
void Relay_Set(Relay_Channel_t channel, bool on);

/** @brief Turn one relay ON. */
void Relay_On(Relay_Channel_t channel);

/** @brief Turn one relay OFF. */
void Relay_Off(Relay_Channel_t channel);

/** @brief Turn all three relays ON. */
void Relay_AllOn(void);

/** @brief Turn all three relays OFF. */
void Relay_AllOff(void);

/** @brief Return true when the selected relay output is HIGH. */
bool Relay_IsOn(Relay_Channel_t channel);

#ifdef __cplusplus
}
#endif

#endif /* MODULE_RELAY_H */
