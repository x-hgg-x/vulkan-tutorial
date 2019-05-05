#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <set>
#include <stddef.h>
#include <stdexcept>

const int WIDTH = 800;
const int HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 1;

const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};
const std::vector<const char *> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;

    static vk::VertexInputBindingDescription getBindingDescription()
    {
        // vk::VertexInputBindingDescription(binding_, stride_, inputRate_)
        return vk::VertexInputBindingDescription(0, sizeof(Vertex), vk::VertexInputRate::eVertex);
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions()
    {
        // vk::VertexInputAttributeDescription(location_, binding_, format_, offset_)
        return {{{0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, pos)},
                 {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)}}};
    }
};

const std::vector<Vertex> vertices = {
    {{-0.5, -0.5}, {1, 0, 0}},
    {{0.5, -0.5}, {0, 1, 0}},
    {{0.5, 0.5}, {0, 0, 1}},
    {{-0.5, 0.5}, {1, 1, 1}}};

const std::vector<uint16_t> indices = {0, 1, 2, 2, 3, 0};

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

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

class HelloVulkan
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

    vk::UniqueInstance instance;
    vk::DispatchLoaderDynamic dldy;
    vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic> debugMessenger;
    vk::UniqueSurfaceKHR surface;

    vk::PhysicalDevice physicalDevice = vk::PhysicalDevice(nullptr);
    vk::UniqueDevice device;

    vk::Queue graphicsQueue;
    vk::Queue presentQueue;

    vk::UniqueSwapchainKHR swapChain;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::Image> swapChainImages;
    std::vector<vk::UniqueImageView> swapChainImageViews;
    std::vector<vk::UniqueFramebuffer> swapChainFramebuffers;

    vk::UniqueRenderPass renderPass;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniquePipelineLayout pipelineLayout;
    std::vector<vk::UniquePipeline> graphicsPipelines;

    vk::UniqueCommandPool commandPool;

    vk::UniqueDeviceMemory vertexBufferMemory;
    vk::UniqueDeviceMemory indexBufferMemory;
    vk::UniqueBuffer vertexBuffer;
    vk::UniqueBuffer indexBuffer;

    std::vector<vk::UniqueDeviceMemory> uniformBuffersMemory;
    std::vector<vk::UniqueBuffer> uniformBuffers;

    std::vector<vk::CommandBuffer> commandBuffers;

    std::vector<vk::UniqueSemaphore> imageAvailableSemaphores;
    std::vector<vk::UniqueSemaphore> renderFinishedSemaphores;
    std::vector<vk::UniqueFence> inFlightFences;
    size_t currentFrame = 0;

    bool framebufferResized = false;

    void initWindow()
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
    {
        auto app = reinterpret_cast<HelloVulkan *>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initVulkan()
    {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window)) {
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, true);
            }
            glfwPollEvents();
            drawFrame();
        }

        device->waitIdle();
    }

    void cleanup()
    {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void recreateSwapChain()
    {
        int width = 0, height = 0;
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        device->waitIdle();
        device->freeCommandBuffers(*commandPool, commandBuffers);

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createUniformBuffers();
        createCommandBuffers();
    }

    void drawFrame()
    {
        device->waitForFences({*inFlightFences[currentFrame]}, VK_TRUE, std::numeric_limits<uint64_t>::max());

        uint32_t imageIndex;
        auto result = device->acquireNextImageKHR(*swapChain, std::numeric_limits<uint64_t>::max(), *imageAvailableSemaphores[currentFrame], {}, &imageIndex);

        if (result == vk::Result::eErrorOutOfDateKHR) {
            recreateSwapChain();
            return;
        } else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        device->resetFences({*inFlightFences[currentFrame]});

        vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};

        updateUniformBuffer(imageIndex);

        // vk::SubmitInfo(waitSemaphoreCount_ pWaitSemaphores_, pWaitDstStageMask_, commandBufferCount_, pCommandBuffers_, signalSemaphoreCount_, pSignalSemaphores_)
        auto submitInfo = vk::SubmitInfo(1, &*imageAvailableSemaphores[currentFrame], waitStages, 1, &commandBuffers[imageIndex], 1, &*renderFinishedSemaphores[currentFrame]);
        graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);

        auto presentInfo = vk::PresentInfoKHR(1, &*renderFinishedSemaphores[currentFrame], 1, &*swapChain, &imageIndex, nullptr);
        result = presentQueue.presentKHR(&presentInfo);

        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void updateUniformBuffer(uint32_t currentImage)
    {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo = {
            glm::rotate(glm::mat4(1), time * glm::radians(90.0f), glm::vec3(0, 0, 1)),
            glm::lookAt(glm::vec3(2, 2, 2.), glm::vec3(0, 0, 0), glm::vec3(0, 0, 1)),
            glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f)};

        ubo.proj[1][1] *= -1;

        auto data = device->mapMemory(*uniformBuffersMemory[currentImage], 0, sizeof(ubo), {});
        memcpy(data, &ubo, sizeof(ubo));
        device->unmapMemory(*uniformBuffersMemory[currentImage]);
    }

    void createSyncObjects()
    {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            imageAvailableSemaphores[i] = device->createSemaphoreUnique({});
            renderFinishedSemaphores[i] = device->createSemaphoreUnique({});
            inFlightFences[i] = device->createFenceUnique({vk::FenceCreateFlagBits::eSignaled});
        }
    }

    void createCommandBuffers()
    {
        // vk::CommandBufferAllocateInfo(commandPool_, level_, commandBufferCount_)
        auto allocInfo = vk::CommandBufferAllocateInfo(*commandPool, vk::CommandBufferLevel::ePrimary, swapChainFramebuffers.size());
        commandBuffers = device->allocateCommandBuffers(allocInfo);

        for (size_t i = 0; i < commandBuffers.size(); i++) {
            // vk::CommandBufferBeginInfo(flags_, pInheritanceInfo_)
            auto beginInfo = vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eSimultaneousUse, nullptr);
            commandBuffers[i].begin(beginInfo);

            auto clearColor = vk::ClearValue(std::array{0.0f, 0.0f, 0.0f, 1.0f});

            // vk::RenderPassBeginInfo(renderPass_, framebuffer_, renderArea_, clearValueCount_, pClearValues_)
            auto renderPassInfo = vk::RenderPassBeginInfo(*renderPass, *swapChainFramebuffers[i], {{0, 0}, swapChainExtent}, 1, &clearColor);
            commandBuffers[i].beginRenderPass(&renderPassInfo, vk::SubpassContents::eInline);
            commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipelines[0]);
            commandBuffers[i].bindVertexBuffers(0, *vertexBuffer, {0});
            commandBuffers[i].bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint16);
            commandBuffers[i].drawIndexed(indices.size(), 1, 0, 0, 0);
            commandBuffers[i].endRenderPass();
            commandBuffers[i].end();
        }
    }

    void createVertexBuffer()
    {
        vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        vk::UniqueDeviceMemory stagingBufferMemory;
        vk::UniqueBuffer stagingBuffer;

        createUniqueBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        auto data = device->mapMemory(*stagingBufferMemory, 0, bufferSize, {});
        memcpy(data, vertices.data(), bufferSize);
        device->unmapMemory(*stagingBufferMemory);

        createUniqueBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);
        copyBuffer(*stagingBuffer, *vertexBuffer, bufferSize);
    }

    void createIndexBuffer()
    {
        vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();
        vk::UniqueDeviceMemory stagingBufferMemory;
        vk::UniqueBuffer stagingBuffer;

        createUniqueBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        auto data = device->mapMemory(*stagingBufferMemory, 0, bufferSize, {});
        memcpy(data, indices.data(), bufferSize);
        device->unmapMemory(*stagingBufferMemory);

        createUniqueBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);
        copyBuffer(*stagingBuffer, *indexBuffer, bufferSize);
    }

    void createUniformBuffers()
    {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(swapChainImages.size());
        uniformBuffersMemory.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            createUniqueBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, uniformBuffers[i], uniformBuffersMemory[i]);
        }
    }

    void createUniqueBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::UniqueBuffer &buffer, vk::UniqueDeviceMemory &bufferMemory)
    {
        // vk::BufferCreateInfo(flags_, size_, usage_, sharingMode_, queueFamilyIndexCount_, pQueueFamilyIndices_)
        auto bufferInfo = vk::BufferCreateInfo({}, size, usage, vk::SharingMode::eExclusive, 0, nullptr);
        buffer = device->createBufferUnique(bufferInfo);

        auto memRequirements = device->getBufferMemoryRequirements(*buffer);
        uint32_t memoryType = findMemoryType(memRequirements.memoryTypeBits, properties);

        // vk::MemoryAllocateInfo(allocationSize_, memoryTypeIndex_)
        auto allocInfo = vk::MemoryAllocateInfo(memRequirements.size, memoryType);
        bufferMemory = device->allocateMemoryUnique(allocInfo);

        device->bindBufferMemory(*buffer, *bufferMemory, 0);
    }

    void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size)
    {
        // vk::CommandBufferAllocateInfo(commandPool_, level_, commandBufferCount_)
        auto allocInfo = vk::CommandBufferAllocateInfo(*commandPool, vk::CommandBufferLevel::ePrimary, 1);
        std::vector<vk::UniqueCommandBuffer> vertexcommandBuffers = device->allocateCommandBuffersUnique(allocInfo);

        // vk::CommandBufferBeginInfo(flags_, pInheritanceInfo_)
        auto beginInfo = vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit, nullptr);
        vertexcommandBuffers[0]->begin(beginInfo);

        // vk::BufferCopy(srcOffset_, DeviceSize dstOffset_, DeviceSize size_)
        auto copyRegion = vk::BufferCopy(0, 0, size);
        vertexcommandBuffers[0]->copyBuffer(srcBuffer, dstBuffer, copyRegion);
        vertexcommandBuffers[0]->end();

        // vk::SubmitInfo(waitSemaphoreCount_ pWaitSemaphores_, pWaitDstStageMask_, commandBufferCount_, pCommandBuffers_, signalSemaphoreCount_, pSignalSemaphores_)
        auto submitInfo = vk::SubmitInfo(0, nullptr, nullptr, 1, &*vertexcommandBuffers[0], 0, nullptr);
        graphicsQueue.submit(submitInfo, nullptr);
        graphicsQueue.waitIdle();
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
    {
        auto memProperties = physicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createCommandPool()
    {
        // vk::CommandPoolCreateInfo(flags_, queueFamilyIndex_)
        auto poolInfo = vk::CommandPoolCreateInfo({}, findQueueFamilies(physicalDevice).graphicsFamily.value());
        commandPool = device->createCommandPoolUnique(poolInfo);
    }

    void createFramebuffers()
    {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            // vk::FramebufferCreateInfo(flags_, renderPass_, attachmentCount_, pAttachments_, width_, height_, layers_)
            auto framebufferInfo = vk::FramebufferCreateInfo({}, *renderPass, 1, &*swapChainImageViews[i], swapChainExtent.width, swapChainExtent.height, 1);
            swapChainFramebuffers[i] = device->createFramebufferUnique(framebufferInfo);
        }
    }

    void createGraphicsPipeline()
    {
        auto vertShaderCode = readFile("shaders/shader.vert.spv");
        auto fragShaderCode = readFile("shaders/shader.frag.spv");

        vk::UniqueShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        vk::UniqueShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        // vk::PipelineShaderStageCreateInfo(flags_, stage_, module_, pName_, pSpecializationInfo_)
        auto vertShaderStageInfo = vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, *vertShaderModule, "main", nullptr);
        auto fragShaderStageInfo = vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, *fragShaderModule, "main", nullptr);
        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        // vk::PipelineVertexInputStateCreateInfo(flags_, vertexBindingDescriptionCount_, pVertexBindingDescriptions_, vertexAttributeDescriptionCount_, pVertexAttributeDescriptions_)
        auto vertexInputInfo = vk::PipelineVertexInputStateCreateInfo({}, 1, &bindingDescription, attributeDescriptions.size(), attributeDescriptions.data());

        // vk::PipelineInputAssemblyStateCreateInfo(flags_, topology_, primitiveRestartEnable_)
        auto inputAssembly = vk::PipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::eTriangleList, VK_FALSE);

        // vk::Viewport(x_, y_, width_, height_, minDepth_, maxDepth_)
        auto viewport = vk::Viewport(0, 0, swapChainExtent.width, swapChainExtent.height, 0, 1);
        // vk::Rect2D(vk::Offset2D(x_, y_), extent_)
        auto scissor = vk::Rect2D({0, 0}, swapChainExtent);
        // vk::PipelineViewportStateCreateInfo(flags_, viewportCount_, pViewports_, scissorCount_, pScissors_)
        auto viewportState = vk::PipelineViewportStateCreateInfo({}, 1, &viewport, 1, &scissor);

        // vk::PipelineRasterizationStateCreateInfo(flags_, depthClampEnable_, rasterizerDiscardEnable_, polygonMode_, cullMode_, frontFace_, depthBiasEnable_, depthBiasConstantFactor_, depthBiasClamp_, depthBiasSlopeFactor_, lineWidth_)
        auto rasterizer = vk::PipelineRasterizationStateCreateInfo({}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack, vk::FrontFace::eClockwise, VK_FALSE, 0, 0, 0, 1);

        auto multisampling = vk::PipelineMultisampleStateCreateInfo();

        // vk::PipelineColorBlendAttachmentState(blendEnable_, srcColorBlendFactor_, dstColorBlendFactor_, colorBlendOp_, srcAlphaBlendFactor_, dstAlphaBlendFactor_, alphaBlendOp_, colorWriteMask_)
        auto colorBlendAttachment = vk::PipelineColorBlendAttachmentState(VK_FALSE,
                                                                          vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
                                                                          vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
                                                                          vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

        // vk::PipelineColorBlendStateCreateInfo(flags_, logicOpEnable_, logicOp_, attachmentCount_, pAttachments_, blendConstants_)
        auto colorBlending = vk::PipelineColorBlendStateCreateInfo({}, VK_FALSE, vk::LogicOp::eCopy, 1, &colorBlendAttachment, {0, 0, 0, 0});

        // vk::PipelineLayoutCreateInfo(flags_, setLayoutCount_, pSetLayouts_, pushConstantRangeCount_, pPushConstantRanges_)
        auto pipelineLayoutInfo = vk::PipelineLayoutCreateInfo({}, 1, &*descriptorSetLayout, 0, nullptr);
        pipelineLayout = device->createPipelineLayoutUnique(pipelineLayoutInfo);

        auto pipelineInfo = vk::GraphicsPipelineCreateInfo({},               // flags_
                                                           2,                // stageCount_
                                                           shaderStages,     // pStages_
                                                           &vertexInputInfo, // pVertexInputState_
                                                           &inputAssembly,   // pInputAssemblyState_
                                                           nullptr,          // pTessellationState_
                                                           &viewportState,   // pViewportState_
                                                           &rasterizer,      // pRasterizationState_
                                                           &multisampling,   // pMultisampleState_
                                                           nullptr,          // pDepthStencilState_
                                                           &colorBlending,   // pColorBlendState_
                                                           nullptr,          // pDynamicState_
                                                           *pipelineLayout,  // layout_
                                                           *renderPass,      // renderPass_
                                                           0,                // subpass_
                                                           nullptr,          // basePipelineHandle_
                                                           -1);              // basePipelineIndex_

        graphicsPipelines = device->createGraphicsPipelinesUnique(vk::PipelineCache(), pipelineInfo);
    }

    std::vector<char> readFile(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error(std::string("failed to open '") + filename + "'!");
        }

        std::vector<char> buffer(file.tellg());
        file.seekg(0);
        file.read(buffer.data(), buffer.size());
        file.close();
        return buffer;
    }

    vk::UniqueShaderModule createShaderModule(const std::vector<char> &code)
    {
        // vk::ShaderModuleCreateInfo(flags_, codeSize_, pCode_)
        auto createInfo = vk::ShaderModuleCreateInfo({}, code.size(), reinterpret_cast<const uint32_t *>(code.data()));
        return device->createShaderModuleUnique(createInfo);
    }

    void createDescriptorSetLayout()
    {
        // vk::DescriptorSetLayoutBinding(binding_, descriptorType_, descriptorCount_, stageFlags_, pImmutableSamplers_)
        auto uboLayoutBinding = vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr);

        // vk::DescriptorSetLayoutCreateInfo(flags_, bindingCount_, pBindings_)
        auto layoutInfo = vk::DescriptorSetLayoutCreateInfo({}, 1, &uboLayoutBinding);
        descriptorSetLayout = device->createDescriptorSetLayoutUnique(layoutInfo);
    }

    void createRenderPass()
    {
        // vk::AttachmentDescription(flags_, format_, samples_, loadOp_, storeOp_, stencilLoadOp_, stencilStoreOp_, initialLayout_, finalLayout_)
        auto colorAttachment = vk::AttachmentDescription({}, swapChainImageFormat, vk::SampleCountFlagBits::e1,
                                                         vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
                                                         vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
                                                         vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR);

        // vk::AttachmentReference(attachment_, layout_)
        auto colorAttachmentRef = vk::AttachmentReference(0, vk::ImageLayout::eColorAttachmentOptimal);

        // vk::SubpassDescription(flags_, pipelineBindPoint_, inputAttachmentCount_, pInputAttachments_, colorAttachmentCount_, pColorAttachments_, pResolveAttachments_, pDepthStencilAttachment_, preserveAttachmentCount_, pPreserveAttachments_)
        auto subpass = vk::SubpassDescription({}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &colorAttachmentRef, nullptr, nullptr, 0, nullptr);

        // vk::SubpassDependency(srcSubpass_, dstSubpass_, srcStageMask_, dstStageMask_, srcAccessMask_, dstAccessMask_, dependencyFlags_)
        auto dependency = vk::SubpassDependency(VK_SUBPASS_EXTERNAL, 0,
                                                vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                                {}, vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite);

        // vk::RenderPassCreateInfo(flags_, attachmentCount_, pAttachments_, subpassCount_, pSubpasses_, dependencyCount_, pDependencies_)
        auto renderPassInfo = vk::RenderPassCreateInfo({}, 1, &colorAttachment, 1, &subpass, 1, &dependency);
        renderPass = device->createRenderPassUnique(renderPassInfo);
    }

    void createImageViews()
    {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            // vk::ImageViewCreateInfo(flags_, image_, viewType_, format_, components_, vk::ImageSubresourceRange(aspectMask_, baseMipLevel_, levelCount_, baseArrayLayer_, layerCount_))
            auto createInfo = vk::ImageViewCreateInfo({}, swapChainImages[i], vk::ImageViewType::e2D, swapChainImageFormat, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
            swapChainImageViews[i] = device->createImageViewUnique(createInfo);
        }
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
                                                     *surface,                                       // surface_
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

        swapChain = device->createSwapchainKHRUnique(createInfo);
        swapChainImages = device->getSwapchainImagesKHR(*swapChain);
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
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            return {std::clamp(static_cast<uint32_t>(width), capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                    std::clamp(static_cast<uint32_t>(height), capabilities.minImageExtent.height, capabilities.maxImageExtent.height)};
        }
    }

    void createLogicalDevice()
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        float queuePriority = 1;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            // vk::DeviceQueueCreateInfo(flags_, queueFamilyIndex_, queueCount_, pQueuePriorities_)
            queueCreateInfos.push_back(vk::DeviceQueueCreateInfo({}, queueFamily, 1, &queuePriority));
        }

        // vk::DeviceCreateInfo(flags_, queueCreateInfoCount_, pQueueCreateInfos_, enabledLayerCount_, ppEnabledLayerNames, enabledExtensionCount_, ppEnabledExtensionNames_, pEnabledFeatures_)
        auto createInfo = vk::DeviceCreateInfo({}, queueCreateInfos.size(), queueCreateInfos.data(), 0, nullptr, deviceExtensions.size(), deviceExtensions.data(), nullptr);

        device = physicalDevice.createDeviceUnique(createInfo);
        graphicsQueue = device->getQueue(indices.graphicsFamily.value(), 0);
        presentQueue = device->getQueue(indices.presentFamily.value(), 0);
    }

    void pickPhysicalDevice()
    {
        for (const auto &device : instance->enumeratePhysicalDevices()) {
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
        return {device.getSurfaceCapabilitiesKHR(*surface), device.getSurfaceFormatsKHR(*surface), device.getSurfacePresentModesKHR(*surface)};
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
            device.getSurfaceSupportKHR(i, *surface, &presentSupport);

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
        VkSurfaceKHR surfaceTmp;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &surfaceTmp) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        // Set the instance as the allocator for the surface for correct order of destruction
        surface = vk::UniqueSurfaceKHR(surfaceTmp, *instance);
    }

    void setupDebugMessenger()
    {
        if (!enableValidationLayers) {
            return;
        }

        auto messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
        auto messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;

        // vk::DebugUtilsMessengerCreateInfoEXT(flags_, messageSeverity_, messageType_, pfnUserCallback_)
        auto createInfo = vk::DebugUtilsMessengerCreateInfoEXT({}, messageSeverity, messageType, debugCallback);
        debugMessenger = instance->createDebugUtilsMessengerEXTUnique(createInfo, nullptr, dldy);
    }

    void createInstance()
    {
        // vk::ApplicationInfo(pApplicationName_, applicationVersion_, pEngineName_, engineVersion_, apiVersion_)
        auto appInfo = vk::ApplicationInfo("Hello Vulkan", VK_MAKE_VERSION(1, 0, 0), "No Engine", VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_1);

        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        uint32_t enabledLayerCount;
        const char *const *ppEnabledLayerNames;
        if (enableValidationLayers) {
            enabledLayerCount = validationLayers.size();
            ppEnabledLayerNames = validationLayers.data();
        } else {
            enabledLayerCount = 0;
            ppEnabledLayerNames = nullptr;
        }

        auto extensions = getRequiredExtensions();
        uint32_t enabledExtensionCount = extensions.size();
        auto ppEnabledExtensionNames = extensions.data();

        // vk::InstanceCreateInfo(flags_, pApplicationInfo_, enabledLayerCount_, ppEnabledLayerNames_, enabledExtensionCount_, ppEnabledExtensionNames_)
        auto createInfo = vk::InstanceCreateInfo({}, &appInfo, enabledLayerCount, ppEnabledLayerNames, enabledExtensionCount, ppEnabledExtensionNames);

        instance = vk::createInstanceUnique(createInfo);
        dldy.init(*instance);
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
    HelloVulkan app;

    try {
        app.run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
