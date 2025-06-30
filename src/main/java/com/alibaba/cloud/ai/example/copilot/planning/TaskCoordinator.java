package com.alibaba.cloud.ai.example.copilot.planning;

import com.alibaba.cloud.ai.example.copilot.service.LlmService;
import com.alibaba.cloud.ai.example.copilot.service.SseService;
import com.alibaba.cloud.ai.example.copilot.template.TemplateBasedProjectGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.SystemPromptTemplate;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/**
 * 任务协调器
 * 负责协调任务规划和执行的整个流程
 * 实现分步执行，每次只执行一个步骤，然后根据结果规划下一步
 */
@Service
public class TaskCoordinator {

    private static final Logger logger = LoggerFactory.getLogger(TaskCoordinator.class);

    private final TaskPlanningService planningService;
    private final LlmService llmService;
    private final SseService sseService;
    private final TemplateBasedProjectGenerator templateGenerator;

    // 存储正在执行的任务
    private final ConcurrentMap<String, TaskPlan> activeTasks = new ConcurrentHashMap<>();

    public TaskCoordinator(TaskPlanningService planningService,
                           LlmService llmService,
                           SseService sseService,
                           TemplateBasedProjectGenerator templateGenerator) {
        this.planningService = planningService;
        this.llmService = llmService;
        this.sseService = sseService;
        this.templateGenerator = templateGenerator;
    }

    /**
     * 开始执行任务
     * @param userRequest 用户请求
     * @param taskId 任务ID
     * @return 任务计划
     */
    public void startTask(String userRequest, String taskId) {
        logger.info("开始执行任务，任务ID: {}", taskId);

        // 检查是否需要使用智能项目生成
        if (shouldUseIntelligentProjectGeneration(userRequest)) {
            logger.info("检测到项目生成需求，使用AI+MCP智能项目生成，任务ID: {}", taskId);
            // 执行基于AI和MCP的智能项目生成
            handleIntelligentProjectGeneration(userRequest, taskId);
        } else {
            // 原有的任务规划流程
            handleRegularTaskPlanning(userRequest, taskId);
        }
    }


    /**
     * 执行单个步骤
     * @param taskPlan 任务计划
     * @param step 步骤
     */
    private void executeStep(String taskId,TaskPlan taskPlan, TaskStep step) {
        logger.info("开始执行步骤，任务ID: {}, 步骤: {}", taskId, step.getStepIndex());

        // 构建提示内容
        String promptContent = String.format(
                """
                步骤索引: %d
                执行要求: %s
                工具名称: %s
                返回结果: %s
                """,
                step.getStepIndex(),
                step.getStepRequirement(),
                step.getToolName() != null ? step.getToolName() : "",
                step.getResult() != null ? step.getResult() : ""
        );

        TaskPlanningPromptBuilder promptBuilder = new TaskPlanningPromptBuilder();
        String systemText = promptBuilder.buildTaskPlanningPrompt(taskPlan, step.getStepIndex(), step.getStepRequirement());
        Message userMessage = new UserMessage(promptContent);
        SystemPromptTemplate systemPromptTemplate = new SystemPromptTemplate(systemText);
        Message systemMessage = systemPromptTemplate.createMessage();
        Prompt prompt = new Prompt(List.of(userMessage, systemMessage));

        // 更新步骤状态为执行中
        step.setStatus("executing");
        step.setStartTime(System.currentTimeMillis());
        sseService.sendTaskUpdate(taskId, taskPlan);

        // 执行计划
        ChatClient chatClient = llmService.getChatClient();
        Flux<String> content = chatClient.prompt(prompt).stream().content();

        // 实时处理流式响应
        StringBuilder resultBuilder = new StringBuilder();
        AtomicLong lastUpdateTime = new AtomicLong(0);
        final long UPDATE_INTERVAL = 300; // 300ms更新间隔

        content.doOnNext(chunk -> {
            // 每收到一个块就追加到结果中
            resultBuilder.append(chunk);
            logger.info("打印返回的块信息：{}", chunk);
            // 实时发送chunk到前端（用于流式显示）
            sseService.sendStepChunkUpdate(taskId, step.getStepIndex(), chunk, false);

            // 节流发送完整任务状态更新
            long currentTime = System.currentTimeMillis();
            if (currentTime - lastUpdateTime.get() >= UPDATE_INTERVAL) {
                lastUpdateTime.set(currentTime);
                step.setResult(resultBuilder.toString());
                sseService.sendTaskUpdate(taskId, taskPlan);
            }
        }).doOnComplete(() -> {
            // 发送步骤完成的chunk标记
            sseService.sendStepChunkUpdate(taskId, step.getStepIndex(), "", true);
        }).blockLast();

        // 步骤执行完成
        String finalResult = resultBuilder.toString();
        step.setStatus("completed");
        step.setEndTime(System.currentTimeMillis());
        step.setResult(finalResult);

        // 发送最终状态更新
        sseService.sendTaskUpdate(taskId, taskPlan);

        logger.info("步骤执行完成，任务ID: {}, 步骤: {}", taskId, step.getStepIndex());
    }

    /**
     * 获取任务状态
     * @param taskId 任务ID
     * @return 任务计划
     */
    public TaskPlan getTaskStatus(String taskId) {
        return activeTasks.get(taskId);
    }

    /**
     * 取消任务
     * @param taskId 任务ID
     * @return 是否成功取消
     */
    public boolean cancelTask(String taskId) {
        TaskPlan taskPlan = activeTasks.get(taskId);
        if (taskPlan != null) {
            taskPlan.setPlanStatus("cancelled");
            sseService.sendTaskUpdate(taskId, taskPlan);
            activeTasks.remove(taskId);
            logger.info("任务已取消，任务ID: {}", taskId);
            return true;
        }
        return false;
    }

    /**
     * 获取所有活跃任务
     * @return 活跃任务映射
     */
    public ConcurrentMap<String, TaskPlan> getActiveTasks() {
        return new ConcurrentHashMap<>(activeTasks);
    }

    /**
     * 清理已完成的任务
     */
    public void cleanupCompletedTasks() {
        activeTasks.entrySet().removeIf(entry -> {
            String status = entry.getValue().getPlanStatus();
            return "completed".equals(status) || "failed".equals(status) || "cancelled".equals(status);
        });
        logger.info("已清理完成的任务，当前活跃任务数: {}", activeTasks.size());
    }

    /**
     * 手动触发下一步规划
     * 用于调试或手动控制执行流程
     * @param taskId 任务ID
     * @param stepResult 步骤执行结果
     * @return 更新后的任务计划
     */
    public TaskPlan triggerNextStep(String taskId, String stepResult) {
        TaskPlan taskPlan = activeTasks.get(taskId);
        if (taskPlan == null) {
            throw new IllegalArgumentException("任务不存在: " + taskId);
        }

        try {
            TaskPlan updatedPlan = planningService.generateNextStep(taskPlan, stepResult);
            activeTasks.put(taskId, updatedPlan);
            sseService.sendTaskUpdate(taskId, updatedPlan);

            logger.info("手动触发下一步规划完成，任务ID: {}", taskId);
            return updatedPlan;

        } catch (Exception e) {
            logger.error("手动触发下一步规划失败，任务ID: {}", taskId, e);
            throw new RuntimeException("触发下一步规划失败: " + e.getMessage(), e);
        }
    }

    /**
     * 重新执行失败的步骤
     * @param taskId 任务ID
     * @param stepIndex 步骤索引
     * @return 执行结果
     */
    public void  retryFailedStep(String taskId, int stepIndex) {
        TaskPlan taskPlan = activeTasks.get(taskId);
        if (taskPlan == null) {
            throw new IllegalArgumentException("任务不存在: " + taskId);
        }

        TaskStep step = taskPlan.getSteps().stream()
                .filter(s -> s.getStepIndex() == stepIndex)
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("步骤不存在: " + stepIndex));

        if (!"failed".equals(step.getStatus())) {
            throw new IllegalStateException("只能重试失败的步骤");
        }

        // 重置步骤状态
        step.setStatus("pending");
        step.setResult(null);
        step.setStartTime(0);
        step.setEndTime(0);

        logger.info("开始重试失败步骤，任务ID: {}, 步骤: {}", taskId, stepIndex);

    }

    /**
     * 检查是否需要使用智能项目生成
     * @param userRequest 用户请求
     * @return 是否需要项目生成
     */
    private boolean shouldUseIntelligentProjectGeneration(String userRequest) {
        String request = userRequest.toLowerCase();

        // 扩展关键词检测，支持更多项目类型
        return request.contains("创建项目") ||
                request.contains("生成项目") ||
                request.contains("新建项目") ||
                request.contains("项目模板") ||
                request.contains("spring boot") ||
                request.contains("vue") ||
                request.contains("聊天应用") ||
                request.contains("ai应用") ||
                request.contains("对话系统") ||
                request.contains("博客系统") ||
                request.contains("商城系统") ||
                request.contains("管理系统") ||
                request.contains("仓库系统") ||
                request.contains("电商") ||
                request.contains("cms") ||
                request.contains("后台管理") ||
                request.contains("api接口") ||
                request.contains("微服务");
    }

    /**
     * 处理基于AI和MCP的智能项目生成
     * @param userRequest 用户请求
     * @param taskId 任务ID
     */
    private void handleIntelligentProjectGeneration(String userRequest, String taskId) {
        // 异步执行智能项目生成
        CompletableFuture.runAsync(() -> {
            try {
                // 1. 创建智能项目生成任务计划
                TaskPlan taskPlan = createIntelligentProjectGenerationPlan(userRequest, taskId);
                activeTasks.put(taskId, taskPlan);
                sseService.sendTaskUpdate(taskId, taskPlan);

                // 2. 执行智能项目生成流程
                executeIntelligentProjectGenerationSteps(taskPlan, userRequest, taskId);

            } catch (Exception e) {
                logger.error("智能项目生成失败，任务ID: {}", taskId, e);
                // 发送错误信息
                sseService.sendTaskUpdate(taskId, createErrorTaskPlan(taskId, e.getMessage()));
            }
        });
    }

    /**
     * 处理常规任务规划流程
     * @param userRequest 用户请求
     * @param taskId 任务ID
     */
    private void handleRegularTaskPlanning(String userRequest, String taskId) {
        CompletableFuture.runAsync(() -> {
            try {
                // 创建初始任务计划
                TaskPlan taskPlan = planningService.createInitialPlan(userRequest, taskId);
                activeTasks.put(taskId, taskPlan);
                sseService.sendTaskUpdate(taskId, taskPlan);

                // 开始执行任务步骤
                executeTaskSteps(taskPlan, taskId);

            } catch (Exception e) {
                logger.error("常规任务规划失败，任务ID: {}", taskId, e);
                sseService.sendTaskUpdate(taskId, createErrorTaskPlan(taskId, e.getMessage()));
            }
        });
    }


    /**
     * 解析用户请求中的项目信息
     */
    private ProjectInfo parseProjectInfo(String userRequest) {
        ProjectInfo info = new ProjectInfo();

        // 使用AI来解析用户请求
        try {
            String prompt = String.format("""
                请分析以下用户请求，提取项目信息：

                用户请求: %s

                请提取以下信息（如果用户没有明确指定，请提供合理的默认值）：
                1. 项目名称（简短的英文名称，适合作为文件夹名）
                2. 项目描述（一句话描述项目功能）
                3. 特殊需求（用户提到的特定功能或要求）

                请按以下格式返回：
                项目名称: [名称]
                项目描述: [描述]
                特殊需求: [需求]
                """, userRequest);

            String response = llmService.getChatClient().prompt()
                    .user(prompt)
                    .call()
                    .content();

            // 解析AI响应
            String[] lines = response.split("\n");
            for (String line : lines) {
                if (line.startsWith("项目名称:")) {
                    info.name = line.substring(5).trim();
                } else if (line.startsWith("项目描述:")) {
                    info.description = line.substring(5).trim();
                } else if (line.startsWith("特殊需求:")) {
                    info.requirements = line.substring(5).trim();
                }
            }

        } catch (Exception e) {
            logger.warn("AI解析项目信息失败，使用默认值", e);
        }

        // 设置默认值
        if (info.name == null || info.name.isEmpty()) {
            info.name = "ai-chat-app";
        }
        if (info.description == null || info.description.isEmpty()) {
            info.description = "基于Spring AI和Vue3的智能聊天应用";
        }
        if (info.requirements == null || info.requirements.isEmpty()) {
            info.requirements = "基础聊天功能";
        }

        return info;
    }

    /**
     * 执行深度定制 - 简化版本，避免重复执行
     * 只提供项目信息和基本指导，不进行复杂的AI调用
     */
    private String executeDeepCustomization(String projectPath, String userRequest, String taskId) {
        try {
            logger.info("开始简化深度定制，项目路径: {}, 用户需求: {}", projectPath, userRequest);

            // 获取项目结构信息
            String projectStructure = getProjectStructure(projectPath);

            // 构建简化的结果信息
            String result = String.format("""
                ## 项目创建完成！

                **项目路径**: %s
                **用户需求**: %s

                ## 当前项目结构
                %s

                ## 下一步操作建议
                1. 项目已基于模板创建并完成基础配置
                2. 您可以直接在项目目录中进行进一步的代码编辑
                3. 后端代码位于: %s/backend/
                4. 前端代码位于: %s/frontend/
                5. 可以根据需求添加新的功能模块

                ## 项目已就绪
                基础的Spring AI + Vue3聊天应用已经创建完成，您可以开始进行具体的功能开发。
                """, projectPath, userRequest, projectStructure, projectPath, projectPath);

            // 通过SSE发送完成信息
            sseService.sendStepChunkUpdate(taskId, 2, result, true);

            logger.info("简化深度定制完成");
            return result;

        } catch (Exception e) {
            logger.error("简化深度定制失败", e);
            return "项目创建完成，但获取详细信息失败: " + e.getMessage();
        }
    }

    /**
     * 构建深度定制的AI提示词 - 简化版本
     */
    private String buildDeepCustomizationPrompt(String projectPath, String userRequest) {
        // 简化版本，不再使用复杂的AI提示词
        return String.format("""
            项目路径: %s
            用户需求: %s

            项目已创建完成，可以进行进一步的开发。
            """, projectPath, userRequest);
    }

    /**
     * 创建智能项目生成任务计划
     * @param userRequest 用户请求
     * @param taskId 任务ID
     * @return 任务计划
     */
    private TaskPlan createIntelligentProjectGenerationPlan(String userRequest, String taskId) {
        TaskPlan taskPlan = new TaskPlan();
        taskPlan.setTaskId(taskId);
        taskPlan.setTitle("AI智能项目生成");
        taskPlan.setDescription("基于用户需求，使用AI和MCP工具智能生成定制化项目");
        taskPlan.setPlanStatus("processing");

        // 步骤1: 分析用户需求
        TaskStep analyzeStep = new TaskStep();
        analyzeStep.setStepIndex(1);
        analyzeStep.setStepRequirement("分析用户需求，确定项目类型和功能特性");
        analyzeStep.setStatus("pending");
        analyzeStep.setToolName("AI分析");
        taskPlan.addStep(analyzeStep);

        // 步骤2: 智能项目架构设计
        TaskStep designStep = new TaskStep();
        designStep.setStepIndex(2);
        designStep.setStepRequirement("基于需求设计项目架构和技术选型");
        designStep.setStatus("pending");
        designStep.setToolName("AI架构设计");
        taskPlan.addStep(designStep);

        // 步骤3: 基于模板智能生成项目
        TaskStep generateStep = new TaskStep();
        generateStep.setStepIndex(3);
        generateStep.setStepRequirement("基于project-template模板，使用MCP工具智能生成定制化项目代码");
        generateStep.setStatus("pending");
        generateStep.setToolName("MCP文件系统工具 + AI代码生成");
        taskPlan.addStep(generateStep);

        // 步骤4: 项目配置和优化
        TaskStep optimizeStep = new TaskStep();
        optimizeStep.setStepIndex(4);
        optimizeStep.setStepRequirement("优化项目配置，添加特定功能模块");
        optimizeStep.setStatus("pending");
        optimizeStep.setToolName("MCP工具 + AI优化");
        taskPlan.addStep(optimizeStep);

        return taskPlan;
    }

    /**
     * 执行智能项目生成步骤
     * @param taskPlan 任务计划
     * @param userRequest 用户请求
     * @param taskId 任务ID
     */
    private void executeIntelligentProjectGenerationSteps(TaskPlan taskPlan, String userRequest, String taskId) {
        try {
            // 步骤1: 分析用户需求
            executeStep(taskId, taskPlan, taskPlan.getSteps().get(0));

            // 步骤2: 智能项目架构设计
            executeStep(taskId, taskPlan, taskPlan.getSteps().get(1));

            // 步骤3: 基于模板智能生成项目
            executeIntelligentProjectGeneration(taskId, taskPlan, taskPlan.getSteps().get(2), userRequest);

            // 步骤4: 项目配置和优化
            executeStep(taskId, taskPlan, taskPlan.getSteps().get(3));

            // 标记任务完成
            taskPlan.setPlanStatus("completed");
            sseService.sendTaskUpdate(taskId, taskPlan);

        } catch (Exception e) {
            logger.error("智能项目生成步骤执行失败，任务ID: {}", taskId, e);
            taskPlan.setPlanStatus("failed");
            sseService.sendTaskUpdate(taskId, taskPlan);
        }
    }

    /**
     * 执行智能项目生成核心步骤
     * @param taskId 任务ID
     * @param taskPlan 任务计划
     * @param step 当前步骤
     * @param userRequest 用户请求
     */
    private void executeIntelligentProjectGeneration(String taskId, TaskPlan taskPlan, TaskStep step, String userRequest) {
        logger.info("开始执行智能项目生成，任务ID: {}", taskId);

        // 更新步骤状态
        step.setStatus("executing");
        step.setStartTime(System.currentTimeMillis());
        sseService.sendTaskUpdate(taskId, taskPlan);

        // 构建智能项目生成提示词
        String intelligentPrompt = buildIntelligentProjectGenerationPrompt(userRequest);

        // 使用AI和MCP工具进行智能项目生成
        ChatClient chatClient = llmService.getChatClient();

        try {
            // 流式执行AI项目生成
            StringBuilder resultBuilder = new StringBuilder();
            AtomicLong lastUpdateTime = new AtomicLong(0);
            final long UPDATE_INTERVAL = 500; // 500ms更新间隔

            Flux<String> content = chatClient.prompt(intelligentPrompt).stream().content();

            content.doOnNext(chunk -> {
                resultBuilder.append(chunk);
                logger.info("AI项目生成进度: {}", chunk);

                // 实时发送进度到前端
                sseService.sendStepChunkUpdate(taskId, step.getStepIndex(), chunk, false);

                // 节流发送完整状态更新
                long currentTime = System.currentTimeMillis();
                if (currentTime - lastUpdateTime.get() >= UPDATE_INTERVAL) {
                    lastUpdateTime.set(currentTime);
                    step.setResult(resultBuilder.toString());
                    sseService.sendTaskUpdate(taskId, taskPlan);
                }
            }).doOnComplete(() -> {
                // 发送完成标记
                sseService.sendStepChunkUpdate(taskId, step.getStepIndex(), "", true);
            }).doOnError(error -> {
                logger.error("流式处理出错: ", error);
            }).subscribe();

            // 等待一段时间让流式处理完成
            try {
                Thread.sleep(2000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                logger.warn("等待流式处理完成时被中断");
            }

            // 步骤执行完成
            String finalResult = resultBuilder.toString();
            step.setStatus("completed");
            step.setEndTime(System.currentTimeMillis());
            step.setResult(finalResult);
            sseService.sendTaskUpdate(taskId, taskPlan);

            logger.info("智能项目生成完成，任务ID: {}", taskId);

        } catch (Exception e) {
            logger.error("智能项目生成失败，任务ID: {}", taskId, e);
            step.setStatus("failed");
            step.setEndTime(System.currentTimeMillis());
            step.setResult("项目生成失败: " + e.getMessage());
            sseService.sendTaskUpdate(taskId, taskPlan);
            throw new RuntimeException("智能项目生成失败", e);
        }
    }

    /**
     * 构建智能项目生成提示词
     * @param userRequest 用户请求
     * @return 提示词
     */
    private String buildIntelligentProjectGenerationPrompt(String userRequest) {
        return String.format("""
            你是一个专业的全栈项目生成助手，需要基于用户需求和现有的project-template模板来智能生成定制化项目。
            
            ## 用户需求
            %s
            
            ## 任务要求
            1. **分析用户需求**：深入理解用户想要创建的项目类型和功能特性
            2. **参考模板结构**：基于现有的project-template目录结构进行仿写和扩展
            3. **智能项目生成**：使用MCP文件系统工具创建新的项目目录和文件
            4. **代码定制化**：根据用户需求生成相应的后端和前端代码
            
            ## 可用工具
            - **MCP文件系统工具**：用于创建目录、文件，读取和写入文件内容
            - **项目模板参考**：project-template目录包含标准的Spring Boot + Vue3项目结构
            
            ## 生成步骤
            1. 首先使用文件系统工具查看project-template的结构
            2. 分析用户需求，确定需要哪些功能模块
            3. 在generated-projects目录下创建新的项目目录
            4. 基于模板结构，生成定制化的项目文件
            5. 根据用户需求修改和扩展代码内容
            
            ## 注意事项
            - 保持项目结构的合理性和可维护性
            - 生成的代码要符合最佳实践
            - 确保前后端代码的一致性和兼容性
            - 添加必要的配置文件和依赖
            
            请开始执行项目生成任务，并实时报告进度。
            """, userRequest);
    }

    /**
     * 执行任务步骤
     * @param taskPlan 任务计划
     * @param taskId 任务ID
     */
    private void executeTaskSteps(TaskPlan taskPlan, String taskId) {
        try {
            for (TaskStep step : taskPlan.getSteps()) {
                if ("pending".equals(step.getStatus())) {
                    executeStep(taskId, taskPlan, step);
                }
            }

            // 标记任务完成
            taskPlan.setPlanStatus("completed");
            sseService.sendTaskUpdate(taskId, taskPlan);

        } catch (Exception e) {
            logger.error("任务步骤执行失败，任务ID: {}", taskId, e);
            taskPlan.setPlanStatus("failed");
            sseService.sendTaskUpdate(taskId, taskPlan);
        }
    }

    /**
     * 获取项目目录结构信息
     * @param projectPath 项目路径
     * @return 格式化的目录结构字符串
     */
    private String getProjectStructure(String projectPath) {
        try {
            StringBuilder structure = new StringBuilder();
            structure.append("```\n");
            structure.append(projectPath).append("/\n");

            Path projectDir = Paths.get(projectPath);
            if (Files.exists(projectDir)) {
                buildDirectoryTree(projectDir, structure, "", 0, 3); // 最多显示3层深度
            } else {
                structure.append("  [项目目录不存在]\n");
            }

            structure.append("```\n");
            return structure.toString();

        } catch (Exception e) {
            logger.warn("获取项目结构失败: {}", projectPath, e);
            return "```\n" + projectPath + "/\n  [无法读取目录结构: " + e.getMessage() + "]\n```\n";
        }
    }

    /**
     * 递归构建目录树结构
     * @param dir 当前目录
     * @param structure 结构字符串构建器
     * @param prefix 前缀字符串
     * @param currentDepth 当前深度
     * @param maxDepth 最大深度
     */
    private void buildDirectoryTree(Path dir, StringBuilder structure, String prefix, int currentDepth, int maxDepth) {
        if (currentDepth >= maxDepth) {
            return;
        }

        try {
            List<Path> entries = Files.list(dir)
                    .filter(path -> !path.getFileName().toString().startsWith(".")) // 过滤隐藏文件
                    .filter(path -> !path.getFileName().toString().equals("target")) // 过滤target目录
                    .filter(path -> !path.getFileName().toString().equals("node_modules")) // 过滤node_modules目录
                    .sorted((a, b) -> {
                        // 目录优先，然后按名称排序
                        boolean aIsDir = Files.isDirectory(a);
                        boolean bIsDir = Files.isDirectory(b);
                        if (aIsDir && !bIsDir) {
                            return -1;
                        }
                        if (!aIsDir && bIsDir) {
                            return 1;
                        }
                        return a.getFileName().toString().compareTo(b.getFileName().toString());
                    })
                    .collect(Collectors.toList());

            for (int i = 0; i < entries.size(); i++) {
                Path entry = entries.get(i);
                boolean isLast = (i == entries.size() - 1);
                String fileName = entry.getFileName().toString();

                if (Files.isDirectory(entry)) {
                    structure.append(prefix)
                            .append(isLast ? "└── " : "├── ")
                            .append(fileName)
                            .append("/\n");

                    String newPrefix = prefix + (isLast ? "    " : "│   ");
                    buildDirectoryTree(entry, structure, newPrefix, currentDepth + 1, maxDepth);
                } else {
                    structure.append(prefix)
                            .append(isLast ? "└── " : "├── ")
                            .append(fileName)
                            .append("\n");
                }
            }

        } catch (IOException e) {
            structure.append(prefix).append("  [读取目录失败: ").append(e.getMessage()).append("]\n");
        }
    }

    /**
     * 使用项目信息增强用户请求
     * @param originalRequest 原始用户请求
     * @param projectInfo 项目信息
     * @return 增强后的用户请求
     */
    private String enhanceUserRequestWithProjectInfo(String originalRequest, String projectInfo) {
        return String.format("""
            ## 原始用户需求
            %s

            ## 项目执行情况
            %s

            ## 继续处理指令
            基于上述已完成的模板项目，请继续根据用户的原始需求进行深度定制和功能开发。
            项目基础框架已就绪，现在可以专注于实现具体的业务功能。
            """, originalRequest, projectInfo);
    }

    /**
     * 使用增强的用户请求继续任务处理
     * @param enhancedUserRequest 增强后的用户请求
     * @param taskId 任务ID
     */
    private void continueTaskProcessingWithEnhancedRequest(String enhancedUserRequest, String taskId) {
        logger.info("继续处理增强后的用户请求，任务ID: {}", taskId);

        try {
            // 获取当前任务计划
            TaskPlan currentPlan = activeTasks.get(taskId);
            if (currentPlan == null) {
                logger.warn("任务计划不存在，创建新的计划，任务ID: {}", taskId);
                currentPlan = new TaskPlan();
                currentPlan.setTaskId(taskId);
                currentPlan.setTitle("继续处理用户需求");
                currentPlan.setDescription("基于已完成的模板项目继续处理用户需求");
            }

            // 更新任务状态为继续处理
            currentPlan.setPlanStatus("continuing");
            sseService.sendTaskUpdate(taskId, currentPlan);

            // 使用增强的请求继续生成任务计划
            TaskPlan continuePlan = planningService.createInitialPlan(enhancedUserRequest, taskId);

            logger.info("输出最终Request: {}", enhancedUserRequest);

            // 合并任务计划（保留已完成的步骤，添加新的步骤）
            if (currentPlan.getSteps() != null) {
                for (TaskStep existingStep : currentPlan.getSteps()) {
                    if (!"completed".equals(existingStep.getStatus())) {
                        break; // 只保留已完成的步骤
                    }
                    continuePlan.getSteps().add(0, existingStep); // 添加到开头
                }
            }

            // 更新任务计划
            activeTasks.put(taskId, continuePlan);
            sseService.sendTaskUpdate(taskId, continuePlan);

            // 执行任务子任务
            executeStepsSequentially(taskId, currentPlan);

        } catch (Exception e) {
            logger.error("继续处理任务失败，任务ID: {}", taskId, e);
            // 发送错误信息
            sseService.sendTaskUpdate(taskId, createErrorTaskPlan(taskId, "继续处理失败: " + e.getMessage()));
        }
    }

    /**
     * 顺序执行任务步骤
     * @param taskPlan 任务计划
     */
    private void executeStepsSequentially(String taskId,TaskPlan taskPlan) {
        for (TaskStep step : taskPlan.getSteps()) {
            try {
                // 设置步骤状态为等待执行
                step.setStatus("waiting");
                sseService.sendTaskUpdate(taskId, taskPlan);

                // 短暂延迟以显示等待状态
                Thread.sleep(500);

                // 执行步骤
                executeStep(taskId,taskPlan, step);

            } catch (Exception e) {
                logger.error("步骤执行失败，任务ID: {}, 步骤: {}", taskId, step.getStepIndex(), e);
                step.setStatus("failed");
                step.setEndTime(System.currentTimeMillis());
                step.setResult("执行失败: " + e.getMessage());
                sseService.sendTaskUpdate(taskId, taskPlan);

                // 如果某个步骤失败，标记整个任务失败
                taskPlan.setPlanStatus("failed");
                sseService.sendTaskUpdate(taskId, taskPlan);
                return;
            }
        }

        // 所有步骤完成，标记任务完成
        taskPlan.setPlanStatus("completed");
        sseService.sendTaskUpdate(taskId, taskPlan);
        logger.info("任务执行完成，任务ID: {}", taskId);
    }

    /**
     * 创建模板项目任务计划
     */
    private TaskPlan createTemplateProjectTaskPlan(String taskId, ProjectInfo projectInfo) {
        TaskPlan taskPlan = new TaskPlan();
        taskPlan.setTaskId(taskId);
        taskPlan.setTitle("基于模板生成项目: " + projectInfo.name);
        taskPlan.setDescription("使用Spring AI + Vue3模板生成项目");
        taskPlan.setPlanStatus("processing");

        // 步骤1: 复制模板项目
        TaskStep copyTemplateStep = new TaskStep();
        copyTemplateStep.setStepIndex(1);
        copyTemplateStep.setStepRequirement("复制基础模板项目");
        copyTemplateStep.setToolName("template_copier");
        copyTemplateStep.setStatus("pending");
        taskPlan.addStep(copyTemplateStep);

        // 步骤2: 基础定制
        TaskStep basicCustomizeStep = new TaskStep();
        basicCustomizeStep.setStepIndex(2);
        basicCustomizeStep.setStepRequirement("基础项目信息定制");
        basicCustomizeStep.setToolName("basic_customizer");
        basicCustomizeStep.setStatus("pending");
        taskPlan.addStep(basicCustomizeStep);

        return taskPlan;
    }

    /**
     * 创建错误任务计划
     */
    private TaskPlan createErrorTaskPlan(String taskId, String errorMessage) {
        TaskPlan errorPlan = new TaskPlan();
        errorPlan.setTaskId(taskId);
        errorPlan.setTitle("任务执行失败");
        errorPlan.setDescription(errorMessage);
        errorPlan.setPlanStatus("failed");
        return errorPlan;
    }

    /**
     * 项目信息内部类
     */
    private static class ProjectInfo {
        String name;
        String description;
        String requirements;
    }
}
