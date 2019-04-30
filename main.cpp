#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>

const int WIDTH = 800;
const int HEIGHT = 600;

const std::vector<const char *> validationLayers = {"VK_LAYER_LUNARG_standard_validation"};
const std::vector<const char *> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() { return graphicsFamily.has_value() && presentFamily.has_value(); }
};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

class HelloTriangleApplication
{
  public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

  private:
    GLFWwindow *window;

    vk::Instance instance;
    vk::DispatchLoaderDynamic dldy;
    vk::DebugUtilsMessengerEXT debugMessenger;
    vk::SurfaceKHR surface;

    vk::PhysicalDevice physicalDevice = vk::PhysicalDevice(nullptr);
    vk::Device device;

    vk::Queue graphicsQueue;
    vk::Queue presentQueue;

    vk::SwapchainKHR swapChain;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;

    void initWindow()
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan()
    {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window)) {
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, true);
            }
            glfwPollEvents();
        }
    }

    void cleanup()
    {
        device.destroy(swapChain, nullptr);
        device.destroy(nullptr);
        instance.destroy(surface, nullptr);

        if (enableValidationLayers) {
            instance.destroyDebugUtilsMessengerEXT(debugMessenger, nullptr, dldy);
        }

        instance.destroy(nullptr);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void createSwapChain()
    {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        auto imageSharingMode = vk::SharingMode::eExclusive;
        auto queueFamilyIndexCount = 0;
        uint32_t *pQueueFamilyIndices = nullptr;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};
        if (indices.graphicsFamily != indices.presentFamily) {
            imageSharingMode = vk::SharingMode::eConcurrent;
            queueFamilyIndexCount = 2;
            pQueueFamilyIndices = queueFamilyIndices;
        }

        auto createInfo = vk::SwapchainCreateInfoKHR({},                                             // flags_
                                                     surface,                                        // surface_
                                                     imageCount,                                     // minImageCount_
                                                     surfaceFormat.format,                           // imageFormat_
                                                     surfaceFormat.colorSpace,                       // imageColorSpace_
                                                     extent,                                         // imageExtent_
                                                     1,                                              // imageArrayLayers_
                                                     vk::ImageUsageFlagBits::eColorAttachment,       // imageUsage_
                                                     imageSharingMode,                               // imageSharingMode_
                                                     queueFamilyIndexCount,                          // queueFamilyIndexCount_
                                                     pQueueFamilyIndices,                            // pQueueFamilyIndices_
                                                     swapChainSupport.capabilities.currentTransform, // preTransform_
                                                     vk::CompositeAlphaFlagBitsKHR::eOpaque,         // compositeAlpha_
                                                     presentMode,                                    // presentMode_
                                                     VK_TRUE,                                        // clipped_
                                                     nullptr);                                       // oldSwapchain_

        swapChain = device.createSwapchainKHR(createInfo, nullptr);
        swapChainImages = device.getSwapchainImagesKHR(swapChain);
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats)
    {
        if (availableFormats.size() == 1 && availableFormats[0].format == vk::Format::eUndefined) {
            return {vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
        }

        for (const auto &availableFormat : availableFormats) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return availableFormat;
            }
        }
        return availableFormats[0];
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes)
    {
        vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;
        for (const auto &availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
            } else if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
                bestMode = availablePresentMode;
            }
        }
        return bestMode;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities)
    {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            return {std::clamp(static_cast<uint32_t>(WIDTH), capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                    std::clamp(static_cast<uint32_t>(HEIGHT), capabilities.minImageExtent.height, capabilities.maxImageExtent.height)};
        }
    }

    void createLogicalDevice()
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            queueCreateInfos.push_back(vk::DeviceQueueCreateInfo({}, queueFamily, 1, &queuePriority));
            // vk::DeviceQueueCreateInfo(flags_, queueFamilyIndex_, queueCount_, pQueuePriorities_)
        }

        auto deviceFeatures = vk::PhysicalDeviceFeatures();
        auto createInfo = vk::DeviceCreateInfo({}, static_cast<uint32_t>(queueCreateInfos.size()), queueCreateInfos.data(), 0, nullptr, static_cast<uint32_t>(deviceExtensions.size()), deviceExtensions.data(), &deviceFeatures);
        // vk::DeviceCreateInfo(flags_, queueCreateInfoCount_, pQueueCreateInfos_, enabledLayerCount_, ppEnabledLayerNames, enabledExtensionCount_, ppEnabledExtensionNames_, pEnabledFeatures_)

        device = physicalDevice.createDevice(createInfo, nullptr);
        graphicsQueue = device.getQueue(indices.graphicsFamily.value(), 0);
        presentQueue = device.getQueue(indices.presentFamily.value(), 0);
    }

    void pickPhysicalDevice()
    {
        for (const auto &device : instance.enumeratePhysicalDevices()) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == vk::PhysicalDevice(nullptr)) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    bool isDeviceSuitable(vk::PhysicalDevice device)
    {
        if (findQueueFamilies(device).isComplete() && checkDeviceExtensionSupport(device)) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            return !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }
        return false;
    }

    SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device)
    {
        return {device.getSurfaceCapabilitiesKHR(surface), device.getSurfaceFormatsKHR(surface), device.getSurfacePresentModesKHR(surface)};
    }

    bool checkDeviceExtensionSupport(vk::PhysicalDevice device)
    {
        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
        for (const auto &extension : device.enumerateDeviceExtensionProperties()) {
            requiredExtensions.erase(extension.extensionName);
        }
        return requiredExtensions.empty();
    }

    QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device)
    {
        QueueFamilyIndices indices;

        int i = 0;
        for (const auto &queueFamily : device.getQueueFamilyProperties()) {
            if (queueFamily.queueCount > 0 && queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            device.getSurfaceSupportKHR(i, surface, &presentSupport);

            if (queueFamily.queueCount > 0 && presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }
            i++;
        }
        return indices;
    }

    void createSurface()
    {
        if (glfwCreateWindowSurface(instance, window, nullptr, reinterpret_cast<VkSurfaceKHR *>(&surface)) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    void setupDebugMessenger()
    {
        if (!enableValidationLayers) {
            return;
        }

        auto messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
        auto messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;

        auto createInfo = vk::DebugUtilsMessengerCreateInfoEXT({}, messageSeverity, messageType, debugCallback);
        // vk::DebugUtilsMessengerCreateInfoEXT(flags_, messageSeverity_, messageType_, pfnUserCallback_)

        debugMessenger = instance.createDebugUtilsMessengerEXT(createInfo, nullptr, dldy);
    }

    void createInstance()
    {
        auto appInfo = vk::ApplicationInfo("Hello Triangle", VK_MAKE_VERSION(1, 0, 0), "No Engine", VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_0);
        // vk::ApplicationInfo(pApplicationName_, applicationVersion_, pEngineName_, engineVersion_, apiVersion_)

        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        uint32_t enabledLayerCount;
        const char *const *ppEnabledLayerNames;
        if (enableValidationLayers) {
            enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            ppEnabledLayerNames = validationLayers.data();
        } else {
            enabledLayerCount = 0;
            ppEnabledLayerNames = nullptr;
        }

        auto extensions = getRequiredExtensions();
        uint32_t enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        auto ppEnabledExtensionNames = extensions.data();

        auto createInfo = vk::InstanceCreateInfo({}, &appInfo, enabledLayerCount, ppEnabledLayerNames, enabledExtensionCount, ppEnabledExtensionNames);
        // vk::InstanceCreateInfo(flags_, pApplicationInfo_, enabledLayerCount_, ppEnabledLayerNames_, enabledExtensionCount_, ppEnabledExtensionNames_)

        instance = vk::createInstance(createInfo, nullptr);
        dldy.init(instance);
    }

    bool checkValidationLayerSupport()
    {
        std::set<std::string> requiredLayers(validationLayers.begin(), validationLayers.end());
        for (const auto &layerProperties : vk::enumerateInstanceLayerProperties()) {
            requiredLayers.erase(layerProperties.layerName);
        }
        return requiredLayers.empty();
    }

    std::vector<const char *> getRequiredExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        return extensions;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData)
    {
        std::cerr << "validation layer: (severity: 0x" << std::setfill('0') << std::setw(4) << std::hex << messageSeverity << ") " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }
};

int main()
{
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
