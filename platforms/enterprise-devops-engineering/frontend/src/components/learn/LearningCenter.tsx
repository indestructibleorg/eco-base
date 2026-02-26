import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BookOpen,
  Code,
  Play,
  Check,
  ChevronLeft,
  Star,
  Trophy,
  Lightbulb,
  RotateCcw,
  SkipForward,
  GraduationCap,
  Terminal,
  FileCode,
  Database,
  Globe,
  Layout,
  Smartphone,
  Sparkles,
  Clock,
  Flame,
  Search,
  BarChart3,
  Loader2,
} from 'lucide-react';
import { RippleButton } from '@/components/ui/ripple-button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import Editor from '@monaco-editor/react';

// Types
interface Course {
  id: string;
  title: string;
  description: string;
  icon: React.ElementType;
  color: string;
  level: 'beginner' | 'intermediate' | 'advanced';
  duration: string;
  lessons: number;
  completedLessons: number;
  language: string;
  tags: string[];
}

interface Lesson {
  id: string;
  title: string;
  type: 'concept' | 'code' | 'quiz' | 'project';
  content: string;
  code?: string;
  language?: string;
  hints: string[];
  expectedOutput?: string;
  completed: boolean;
}

interface Achievement {
  id: string;
  title: string;
  description: string;
  icon: React.ElementType;
  unlocked: boolean;
  unlockedAt?: Date;
}

interface UserProgress {
  streak: number;
  totalXP: number;
  level: number;
  completedCourses: number;
  completedLessons: number;
}

// Mock Data
const courses: Course[] = [
  {
    id: 'python-basics',
    title: 'Python 基础入门',
    description: '从零开始学习 Python，掌握编程基础概念',
    icon: Terminal,
    color: '#3776AB',
    level: 'beginner',
    duration: '4 周',
    lessons: 24,
    completedLessons: 8,
    language: 'Python',
    tags: ['编程入门', '数据分析', '自动化'],
  },
  {
    id: 'web-frontend',
    title: 'Web 前端开发',
    description: '学习 HTML、CSS、JavaScript，构建现代化网页',
    icon: Globe,
    color: '#E34F26',
    level: 'beginner',
    duration: '6 周',
    lessons: 36,
    completedLessons: 12,
    language: 'JavaScript',
    tags: ['网页开发', '响应式设计', '交互'],
  },
  {
    id: 'react-fundamentals',
    title: 'React 基础',
    description: '掌握 React 框架，构建动态用户界面',
    icon: Layout,
    color: '#61DAFB',
    level: 'intermediate',
    duration: '5 周',
    lessons: 30,
    completedLessons: 0,
    language: 'TypeScript',
    tags: ['前端框架', '组件开发', '状态管理'],
  },
  {
    id: 'sql-database',
    title: 'SQL 数据库',
    description: '学习数据库设计和 SQL 查询语言',
    icon: Database,
    color: '#336791',
    level: 'beginner',
    duration: '3 周',
    lessons: 18,
    completedLessons: 0,
    language: 'SQL',
    tags: ['数据库', '查询', '数据分析'],
  },
  {
    id: 'typescript-basics',
    title: 'TypeScript 入门',
    description: '为 JavaScript 添加类型系统，提升代码质量',
    icon: FileCode,
    color: '#3178C6',
    level: 'intermediate',
    duration: '4 周',
    lessons: 24,
    completedLessons: 0,
    language: 'TypeScript',
    tags: ['类型系统', '大型项目', '代码质量'],
  },
  {
    id: 'mobile-react-native',
    title: 'React Native 移动开发',
    description: '使用 React 构建 iOS 和 Android 应用',
    icon: Smartphone,
    color: '#61DAFB',
    level: 'advanced',
    duration: '8 周',
    lessons: 48,
    completedLessons: 0,
    language: 'JavaScript',
    tags: ['移动开发', '跨平台', 'App'],
  },
];

const currentLesson: Lesson = {
  id: 'lesson-1',
  title: 'Hello World - 你的第一行代码',
  type: 'code',
  content: `欢迎来到编程世界！让我们从最简单的程序开始。

**什么是 "Hello World"？**
"Hello World" 是程序员的传统，代表你写出的第一个程序。它会在屏幕上显示 "Hello, World!" 这句话。

**任务：**
在右侧的代码编辑器中，修改代码让它显示 "Hello, Python!" 而不是 "Hello, World!"。

**提示：**
- 找到引号内的文字
- 修改为你想要的文字
- 点击 "运行" 按钮查看结果`,
  code: `print("Hello, World!")`,
  language: 'python',
  hints: [
    '查看 print 函数括号内的内容',
    '把 "World" 改成 "Python"',
    '注意保持引号不变',
  ],
  expectedOutput: 'Hello, Python!',
  completed: false,
};

const achievements: Achievement[] = [
  { id: '1', title: '初学者', description: '完成第一堂课', icon: Star, unlocked: true, unlockedAt: new Date() },
  { id: '2', title: '连续学习', description: '连续学习 3 天', icon: Flame, unlocked: true, unlockedAt: new Date() },
  { id: '3', title: '代码大师', description: '完成 10 个代码练习', icon: Code, unlocked: false },
  { id: '4', title: '完美通关', description: '不借助提示完成课程', icon: Trophy, unlocked: false },
  { id: '5', title: '知识渊博', description: '完成 5 门课程', icon: GraduationCap, unlocked: false },
];

const userProgress: UserProgress = {
  streak: 5,
  totalXP: 1250,
  level: 3,
  completedCourses: 1,
  completedLessons: 20,
};

export function LearningCenter() {
  const [activeTab, setActiveTab] = useState('courses');
  const [selectedCourse, setSelectedCourse] = useState<Course | null>(null);
  const [currentLessonState] = useState<Lesson>(currentLesson);
  const [userCode, setUserCode] = useState(currentLesson.code || '');
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [showHint, setShowHint] = useState(false);
  const [currentHintIndex, setCurrentHintIndex] = useState(0);
  const [showCelebration, setShowCelebration] = useState(false);

  // Run code
  const runCode = useCallback(() => {
    setIsRunning(true);
    setOutput('');

    // Simulate code execution
    setTimeout(() => {
      if (userCode.includes('Hello, Python!')) {
        setOutput('Hello, Python!\n\n✅ 太棒了！你成功完成了第一个编程任务！');
        setShowCelebration(true);
        setTimeout(() => setShowCelebration(false), 3000);
      } else if (userCode.includes('Hello, World!')) {
        setOutput('Hello, World!\n\n💡 提示：试着把 "World" 改成 "Python"');
      } else {
        setOutput('🤔 输出：\n' + userCode + '\n\n💡 提示：使用 print() 函数输出文字');
      }
      setIsRunning(false);
    }, 1000);
  }, [userCode]);

  // Show next hint
  const showNextHint = () => {
    if (currentHintIndex < currentLessonState.hints.length - 1) {
      setCurrentHintIndex((prev) => prev + 1);
    }
    setShowHint(true);
  };

  // Render course card
  const renderCourseCard = (course: Course) => (
    <motion.div
      key={course.id}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={() => setSelectedCourse(course)}
      className="bg-gray-50 border border-gray-200 rounded-xl p-5 cursor-pointer hover:border-[#FFD700]/50 transition-colors"
    >
      <div className="flex items-start justify-between mb-4">
        <div
          className="w-14 h-14 rounded-xl flex items-center justify-center"
          style={{ backgroundColor: `${course.color}20` }}
        >
          <course.icon className="w-7 h-7" style={{ color: course.color }} />
        </div>
        <Badge
          className={cn(
            'border-0 text-[10px]',
            course.level === 'beginner' && 'bg-green-100 text-green-600',
            course.level === 'intermediate' && 'bg-yellow-100 text-yellow-600',
            course.level === 'advanced' && 'bg-red-100 text-red-600'
          )}
        >
          {course.level === 'beginner' ? '入门' : course.level === 'intermediate' ? '进阶' : '高级'}
        </Badge>
      </div>
      <h3 className="font-semibold text-gray-900 mb-2">{course.title}</h3>
      <p className="text-sm text-gray-500 mb-4">{course.description}</p>
      <div className="flex items-center gap-4 text-xs text-gray-600 mb-3">
        <span className="flex items-center gap-1">
          <Clock className="w-3 h-3" />
          {course.duration}
        </span>
        <span className="flex items-center gap-1">
          <BookOpen className="w-3 h-3" />
          {course.lessons} 课
        </span>
      </div>
      <div className="flex flex-wrap gap-1">
        {course.tags.slice(0, 2).map((tag, i) => (
          <Badge key={i} className="bg-white text-gray-600 border-0 text-[9px]">
            {tag}
          </Badge>
        ))}
      </div>
      {course.completedLessons > 0 && (
        <div className="mt-4">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-gray-500">进度</span>
            <span className="text-[#FFD700]">{Math.round((course.completedLessons / course.lessons) * 100)}%</span>
          </div>
          <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-[#FFD700] transition-all"
              style={{ width: `${(course.completedLessons / course.lessons) * 100}%` }}
            />
          </div>
        </div>
      )}
    </motion.div>
  );

  return (
    <div className="h-full flex bg-white">
      {/* Sidebar */}
      <div className="w-72 border-r border-gray-200 flex flex-col bg-white">
        {/* User Progress */}
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-[#FFD700] to-[#FFA500] flex items-center justify-center">
              <GraduationCap className="w-6 h-6 text-black" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">学习进度</h3>
              <p className="text-xs text-gray-500">Lv.{userProgress.level} 学习者</p>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <div className="bg-gray-50 rounded-lg p-2 text-center">
              <div className="text-lg font-bold text-[#FFD700]">{userProgress.streak}</div>
              <div className="text-[10px] text-gray-500">连续天数</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-2 text-center">
              <div className="text-lg font-bold text-[#10B981]">{userProgress.totalXP}</div>
              <div className="text-[10px] text-gray-500">XP</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-2 text-center">
              <div className="text-lg font-bold text-[#8B5CF6]">{userProgress.completedLessons}</div>
              <div className="text-[10px] text-gray-500">已完成</div>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <div className="flex-1 py-4 px-3">
          <nav className="space-y-1">
            {[
              { id: 'courses', label: '课程', icon: BookOpen },
              { id: 'practice', label: '练习', icon: Code },
              { id: 'achievements', label: '成就', icon: Trophy },
              { id: 'leaderboard', label: '排行榜', icon: BarChart3 },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={cn(
                  'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all',
                  activeTab === item.id
                    ? 'bg-[#FFD700]/10 text-[#FFD700]'
                    : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                )}
              >
                <item.icon className="w-5 h-5" />
                <span className="font-medium text-sm">{item.label}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Daily Goal */}
        <div className="p-4 border-t border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-900">今日目标</span>
            <span className="text-xs text-[#FFD700]">2/3 完成</span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div className="h-full bg-gradient-to-r from-[#FFD700] to-[#FFA500] w-[66%]" />
          </div>
          <div className="mt-3 space-y-2">
            <div className="flex items-center gap-2 text-xs">
              <Check className="w-4 h-4 text-green-500" />
              <span className="text-gray-600">完成 1 节课</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <Check className="w-4 h-4 text-green-500" />
              <span className="text-gray-600">获得 50 XP</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <div className="w-4 h-4 rounded-full border border-gray-400" />
              <span className="text-gray-500">连续学习 6 天</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="h-14 border-b border-gray-200 flex items-center justify-between px-6">
          <h1 className="text-xl font-semibold text-gray-900">
            {activeTab === 'courses' && '课程中心'}
            {activeTab === 'practice' && '代码练习'}
            {activeTab === 'achievements' && '成就系统'}
            {activeTab === 'leaderboard' && '排行榜'}
          </h1>
          <div className="flex items-center gap-2">
            <RippleButton variant="outline" size="sm" className="gap-2 border-gray-300">
              <Search className="w-4 h-4" />
              搜索
            </RippleButton>
            <RippleButton size="sm" className="gap-2 bg-[#FFD700] text-black hover:bg-[#FFC700]">
              <Sparkles className="w-4 h-4" />
              继续学习
            </RippleButton>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-hidden">
          {/* Courses Tab */}
          {activeTab === 'courses' && !selectedCourse && (
            <ScrollArea className="h-full p-6">
              <div className="max-w-5xl mx-auto">
                {/* Featured Course */}
                <div className="mb-8">
                  <div className="bg-gradient-to-r from-[#FFD700] to-[#FFA500] rounded-2xl p-6">
                    <div className="flex items-start justify-between">
                      <div>
                        <Badge className="bg-white/30 text-black border-0 mb-3">推荐课程</Badge>
                        <h2 className="text-2xl font-bold text-black mb-2">Python 基础入门</h2>
                        <p className="text-black/70 max-w-lg mb-4">
                          从零开始学习 Python，掌握编程基础概念。适合完全没有编程经验的初学者。
                        </p>
                        <div className="flex gap-2">
                          <RippleButton className="bg-white text-black hover:bg-gray-100">
                            <Play className="w-4 h-4 mr-2" />
                            开始学习
                          </RippleButton>
                          <RippleButton variant="outline" className="border-black/30 text-black hover:bg-black/10">
                            查看详情
                          </RippleButton>
                        </div>
                      </div>
                      <div className="w-24 h-24 bg-white/30 rounded-2xl flex items-center justify-center">
                        <Terminal className="w-12 h-12 text-black" />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Course Categories */}
                <div className="flex items-center gap-2 mb-6">
                  {['全部', '入门', '进阶', '高级'].map((cat, i) => (
                    <button
                      key={cat}
                      className={cn(
                        'px-4 py-2 rounded-lg text-sm transition-colors',
                        i === 0
                          ? 'bg-[#FFD700] text-black'
                          : 'bg-gray-50 text-gray-600 hover:bg-gray-100'
                      )}
                    >
                      {cat}
                    </button>
                  ))}
                </div>

                {/* Course Grid */}
                <div className="grid grid-cols-3 gap-4">
                  {courses.map(renderCourseCard)}
                </div>
              </div>
            </ScrollArea>
          )}

          {/* Lesson View */}
          {activeTab === 'courses' && selectedCourse && (
            <div className="h-full flex">
              {/* Lesson Content */}
              <div className="flex-1 flex flex-col">
                {/* Lesson Header */}
                <div className="h-14 border-b border-gray-200 flex items-center justify-between px-4">
                  <div className="flex items-center gap-3">
                    <RippleButton
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-gray-500"
                      onClick={() => setSelectedCourse(null)}
                    >
                      <ChevronLeft className="w-5 h-5" />
                    </RippleButton>
                    <div>
                      <h2 className="font-semibold text-gray-900">{currentLessonState.title}</h2>
                      <p className="text-xs text-gray-500">{selectedCourse.title}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <RippleButton variant="outline" size="sm" className="gap-2 border-gray-300">
                      <Lightbulb className="w-4 h-4" />
                      提示
                    </RippleButton>
                    <RippleButton size="sm" className="gap-2 bg-[#FFD700] text-black hover:bg-[#FFC700]">
                      <Check className="w-4 h-4" />
                      完成
                    </RippleButton>
                  </div>
                </div>

                {/* Lesson Body */}
                <div className="flex-1 flex overflow-hidden">
                  {/* Instructions */}
                  <ScrollArea className="w-1/2 p-6 border-r border-gray-200">
                    <div className="prose max-w-none">
                      <div className="whitespace-pre-wrap text-gray-700">
                        {currentLessonState.content.split('\n\n').map((paragraph, i) => {
                          if (paragraph.startsWith('**') && paragraph.endsWith('**')) {
                            return <h3 key={i} className="text-lg font-semibold text-gray-900 mt-6 mb-3">{paragraph.replace(/\*\*/g, '')}</h3>;
                          }
                          if (paragraph.startsWith('- ')) {
                            return (
                              <ul key={i} className="list-disc list-inside space-y-1 text-gray-600 mb-4">
                                {paragraph.split('\n').map((item, j) => (
                                  <li key={j}>{item.replace('- ', '')}</li>
                                ))}
                              </ul>
                            );
                          }
                          return <p key={i} className="mb-4">{paragraph}</p>;
                        })}
                      </div>
                    </div>

                    {/* Hints */}
                    {showHint && (
                      <div className="mt-6 p-4 bg-[#FFD700]/10 border border-[#FFD700]/30 rounded-xl">
                        <div className="flex items-center gap-2 mb-2">
                          <Lightbulb className="w-4 h-4 text-[#FFD700]" />
                          <span className="font-medium text-[#FFD700]">提示 {currentHintIndex + 1}/{currentLessonState.hints.length}</span>
                        </div>
                        <p className="text-sm text-gray-600">{currentLessonState.hints[currentHintIndex]}</p>
                        {currentHintIndex < currentLessonState.hints.length - 1 && (
                          <button
                            onClick={showNextHint}
                            className="mt-2 text-sm text-[#FFD700] hover:underline"
                          >
                            下一个提示 →
                          </button>
                        )}
                      </div>
                    )}
                  </ScrollArea>

                  {/* Code Editor */}
                  <div className="flex-1 flex flex-col">
                    <div className="flex-1">
                      <Editor
                        height="100%"
                        language={currentLessonState.language}
                        value={userCode}
                        onChange={(value) => value !== undefined && setUserCode(value)}
                        theme="vs-light"
                        options={{
                          minimap: { enabled: false },
                          fontSize: 14,
                          lineNumbers: 'on',
                          roundedSelection: false,
                          scrollBeyondLastLine: false,
                          automaticLayout: true,
                          fontFamily: 'JetBrains Mono, Fira Code, monospace',
                        }}
                      />
                    </div>

                    {/* Output Console */}
                    <div className="h-40 border-t border-gray-200 bg-gray-50">
                      <div className="h-8 flex items-center justify-between px-4 border-b border-gray-200">
                        <div className="flex items-center gap-2">
                          <Terminal className="w-4 h-4 text-gray-500" />
                          <span className="text-sm text-gray-600">输出</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <RippleButton
                            variant="ghost"
                            size="sm"
                            className="h-6 text-xs gap-1 text-gray-500"
                            onClick={showNextHint}
                          >
                            <Lightbulb className="w-3 h-3" />
                            提示
                          </RippleButton>
                          <RippleButton
                            variant="ghost"
                            size="sm"
                            className="h-6 text-xs gap-1 text-gray-500"
                            onClick={() => setUserCode(currentLessonState.code || '')}
                          >
                            <RotateCcw className="w-3 h-3" />
                            重置
                          </RippleButton>
                        </div>
                      </div>
                      <ScrollArea className="h-[calc(100%-32px)] p-4 font-mono text-sm">
                        {output ? (
                          <pre className="text-gray-800 whitespace-pre-wrap">{output}</pre>
                        ) : (
                          <p className="text-gray-500">点击 "运行" 查看输出结果...</p>
                        )}
                      </ScrollArea>
                    </div>

                    {/* Action Bar */}
                    <div className="h-14 border-t border-gray-200 flex items-center justify-between px-4">
                      <div className="flex items-center gap-2">
                        <RippleButton variant="outline" size="sm" className="gap-2 border-gray-300">
                          <SkipForward className="w-4 h-4" />
                          跳过
                        </RippleButton>
                      </div>
                      <RippleButton
                        size="sm"
                        className="gap-2 bg-[#FFD700] text-black hover:bg-[#FFC700]"
                        onClick={runCode}
                        disabled={isRunning}
                      >
                        {isRunning ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <Play className="w-4 h-4" />
                        )}
                        运行
                      </RippleButton>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Achievements Tab */}
          {activeTab === 'achievements' && (
            <ScrollArea className="h-full p-6">
              <div className="max-w-4xl mx-auto">
                <div className="grid grid-cols-3 gap-4">
                  {achievements.map((achievement) => (
                    <div
                      key={achievement.id}
                      className={cn(
                        'p-5 rounded-xl border transition-all',
                        achievement.unlocked
                          ? 'bg-gray-50 border-[#FFD700]/50'
                          : 'bg-white border-gray-200 opacity-60'
                      )}
                    >
                      <div
                        className={cn(
                          'w-14 h-14 rounded-xl flex items-center justify-center mb-4',
                          achievement.unlocked ? 'bg-[#FFD700]/20' : 'bg-gray-100'
                        )}
                      >
                        <achievement.icon
                          className={cn(
                            'w-7 h-7',
                            achievement.unlocked ? 'text-[#FFD700]' : 'text-gray-500'
                          )}
                        />
                      </div>
                      <h3 className="font-semibold text-gray-900 mb-1">{achievement.title}</h3>
                      <p className="text-sm text-gray-500">{achievement.description}</p>
                      {achievement.unlocked && achievement.unlockedAt && (
                        <p className="text-xs text-[#FFD700] mt-2">
                          解锁于 {achievement.unlockedAt.toLocaleDateString()}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </ScrollArea>
          )}

          {/* Leaderboard Tab */}
          {activeTab === 'leaderboard' && (
            <ScrollArea className="h-full p-6">
              <div className="max-w-2xl mx-auto">
                <div className="bg-gray-50 border border-gray-200 rounded-xl overflow-hidden">
                  <div className="p-4 border-b border-gray-200">
                    <div className="flex items-center justify-between">
                      <h3 className="font-semibold text-gray-900">本周排行榜</h3>
                      <div className="flex items-center gap-2">
                        <RippleButton variant="ghost" size="sm" className="text-gray-600">全球</RippleButton>
                        <RippleButton variant="ghost" size="sm" className="text-gray-600">好友</RippleButton>
                      </div>
                    </div>
                  </div>
                  <div className="divide-y divide-gray-200">
                    {[
                      { rank: 1, name: 'Alex Chen', xp: 5230, avatar: 'A', streak: 12 },
                      { rank: 2, name: 'Sarah Wang', xp: 4890, avatar: 'S', streak: 8 },
                      { rank: 3, name: 'Mike Liu', xp: 4560, avatar: 'M', streak: 15 },
                      { rank: 4, name: 'Emma Zhang', xp: 4120, avatar: 'E', streak: 6 },
                      { rank: 5, name: 'David Li', xp: 3890, avatar: 'D', streak: 9 },
                    ].map((user, i) => (
                      <div
                        key={i}
                        className={cn(
                          'flex items-center gap-4 p-4',
                          user.rank <= 3 && 'bg-[#FFD700]/5'
                        )}
                      >
                        <div
                          className={cn(
                            'w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm',
                            user.rank === 1 && 'bg-yellow-100 text-yellow-600',
                            user.rank === 2 && 'bg-gray-100 text-gray-600',
                            user.rank === 3 && 'bg-orange-100 text-orange-600',
                            user.rank > 3 && 'text-gray-500'
                          )}
                        >
                          {user.rank}
                        </div>
                        <div className="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center text-gray-700 font-medium">
                          {user.avatar}
                        </div>
                        <div className="flex-1">
                          <div className="font-medium text-gray-900">{user.name}</div>
                          <div className="text-xs text-gray-500">{user.xp} XP</div>
                        </div>
                        <div className="flex items-center gap-1 text-[#FFD700]">
                          <Flame className="w-4 h-4" />
                          <span className="text-sm">{user.streak}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </ScrollArea>
          )}
        </div>
      </div>

      {/* Celebration Overlay */}
      <AnimatePresence>
        {showCelebration && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none bg-white/80"
          >
            <div className="text-center">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                exit={{ scale: 0 }}
                className="w-24 h-24 rounded-full bg-gradient-to-br from-[#FFD700] to-[#FFA500] flex items-center justify-center mx-auto mb-4"
              >
                <Trophy className="w-12 h-12 text-black" />
              </motion.div>
              <motion.h2
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                exit={{ y: 20, opacity: 0 }}
                className="text-2xl font-bold text-gray-900"
              >
                太棒了！
              </motion.h2>
              <motion.p
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.1 }}
                className="text-gray-500"
              >
                你成功完成了这个练习！
              </motion.p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
