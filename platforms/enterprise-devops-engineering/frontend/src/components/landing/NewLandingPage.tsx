import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import {
  Code,
  Terminal,
  Cpu,
  Globe,
  Shield,
  Check,
  ArrowRight,
  Play,
  Star,
  Database,
  Sparkles,
  Menu,
  X,
} from 'lucide-react';
import { RippleButton } from '@/components/ui/ripple-button';
import { cn } from '@/lib/utils';

// Features data
const features = [
  { icon: Code, key: 'aiCodeGen', color: '#FFD700' },
  { icon: Terminal, key: 'cloudIDE', color: '#1E88E5' },
  { icon: Cpu, key: 'aiAgents', color: '#10B981' },
  { icon: Globe, key: 'oneClickDeploy', color: '#F26207' },
  { icon: Shield, key: 'codeSecurity', color: '#8B5CF6' },
  { icon: Database, key: 'databaseMgmt', color: '#06B6D4' },
];

// Stats data
const stats = [
  { value: '100K+', key: 'developers' },
  { value: '500+', key: 'enterpriseClients' },
  { value: '99.9%', key: 'uptime' },
  { value: '24/7', key: 'support' },
];

// Pricing plans
const pricingPlans = [
  { key: 'free', highlighted: false },
  { key: 'pro', highlighted: true },
  { key: 'enterprise', highlighted: false },
];

export function NewLandingPage() {
  const { t, i18n } = useTranslation();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const navItems = [
    { label: t('landing.hero.products'), href: '#products' },
    { label: t('landing.hero.features'), href: '#features' },
    { label: t('landing.hero.pricing'), href: '#pricing' },
    { label: t('landing.hero.docs'), href: '#docs' },
  ];

  const toggleLanguage = () => {
    const newLang = i18n.language === 'en' ? 'zh-TW' : 'en';
    i18n.changeLanguage(newLang);
  };

  return (
    <div className="min-h-screen bg-white overflow-y-auto">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white/95 backdrop-blur-md border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#FFD700] to-[#FFA500] flex items-center justify-center">
                <Code className="w-6 h-6 text-black" />
              </div>
              <span className="text-xl font-bold text-gray-900">CodeAI</span>
            </div>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center gap-8">
              {navItems.map((item) => (
                <a
                  key={item.label}
                  href={item.href}
                  className="text-gray-600 hover:text-gray-900 font-medium transition-colors"
                >
                  {item.label}
                </a>
              ))}
            </div>

            {/* CTA Buttons */}
            <div className="hidden md:flex items-center gap-3">
              <button
                onClick={toggleLanguage}
                className="text-sm text-gray-600 hover:text-gray-900 font-medium"
              >
                {i18n.language === 'en' ? '繁體中文' : 'English'}
              </button>
              <RippleButton variant="outline" className="border-gray-300 text-gray-700">
                {t('common.login')}
              </RippleButton>
              <RippleButton className="bg-[#FFD700] text-black hover:bg-[#FFC700] font-semibold">
                {t('common.signup')}
              </RippleButton>
            </div>

            {/* Mobile Menu Button */}
            <button
              className="md:hidden p-2"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            >
              {isMobileMenuOpen ? (
                <X className="w-6 h-6 text-gray-700" />
              ) : (
                <Menu className="w-6 h-6 text-gray-700" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="md:hidden bg-white border-t border-gray-100"
          >
            <div className="px-4 py-4 space-y-3">
              {navItems.map((item) => (
                <a
                  key={item.label}
                  href={item.href}
                  className="block py-2 text-gray-600 hover:text-gray-900 font-medium"
                >
                  {item.label}
                </a>
              ))}
              <button
                onClick={toggleLanguage}
                className="block py-2 text-gray-600 hover:text-gray-900 font-medium"
              >
                {i18n.language === 'en' ? '繁體中文' : 'English'}
              </button>
              <div className="pt-3 space-y-2">
                <RippleButton variant="outline" className="w-full border-gray-300">
                  {t('common.login')}
                </RippleButton>
                <RippleButton className="w-full bg-[#FFD700] text-black font-semibold">
                  {t('common.signup')}
                </RippleButton>
              </div>
            </div>
          </motion.div>
        )}
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left Content */}
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
            >
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-yellow-50 rounded-full mb-6">
                <Sparkles className="w-4 h-4 text-[#FFD700]" />
                <span className="text-sm font-medium text-gray-700">
                  {t('landing.hero.badge')}
                </span>
              </div>
              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 leading-tight mb-6">
                {t('landing.hero.title1')}
                <span className="text-[#FFD700]">{t('landing.hero.titleHighlight1')}</span>
                <br />
                {t('landing.hero.title2')}
                <span className="text-[#1E88E5]">{t('landing.hero.titleHighlight2')}</span>
              </h1>
              <p className="text-lg text-gray-600 mb-8 max-w-lg">
                {t('landing.hero.description')}
              </p>
              <div className="flex flex-wrap gap-4">
                <RippleButton
                  size="lg"
                  className="bg-[#FFD700] text-black hover:bg-[#FFC700] font-semibold px-8"
                >
                  <Play className="w-5 h-5 mr-2" />
                  {t('landing.hero.startFree')}
                </RippleButton>
                <RippleButton
                  size="lg"
                  variant="outline"
                  className="border-gray-300 text-gray-700 px-8"
                >
                  {t('landing.hero.watchDemo')}
                  <ArrowRight className="w-5 h-5 ml-2" />
                </RippleButton>
              </div>

              {/* Trust Badges */}
              <div className="mt-10 flex items-center gap-6 text-sm text-gray-500">
                <div className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-green-500" />
                  <span>{t('landing.hero.noCreditCard')}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-green-500" />
                  <span>{t('landing.hero.freeTrial')}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-green-500" />
                  <span>{t('landing.hero.cancelAnytime')}</span>
                </div>
              </div>
            </motion.div>

            {/* Right Content - Code Preview */}
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="relative"
            >
              <div className="bg-gray-900 rounded-2xl shadow-2xl overflow-hidden">
                {/* Window Header */}
                <div className="flex items-center gap-2 px-4 py-3 bg-gray-800">
                  <div className="w-3 h-3 rounded-full bg-red-500" />
                  <div className="w-3 h-3 rounded-full bg-yellow-500" />
                  <div className="w-3 h-3 rounded-full bg-green-500" />
                  <span className="ml-4 text-sm text-gray-400">main.py</span>
                </div>
                {/* Code Content */}
                <div className="p-6 font-mono text-sm">
                  <div className="text-gray-300">
                    <span className="text-purple-400">def</span>
                    <span className="text-blue-400"> calculate_fibonacci</span>
                    <span className="text-gray-400">(n):</span>
                  </div>
                  <div className="text-gray-300 ml-4">
                    <span className="text-purple-400">if</span>
                    <span className="text-gray-400"> n &lt;= </span>
                    <span className="text-orange-400">1</span>
                    <span className="text-gray-400">:</span>
                  </div>
                  <div className="text-gray-300 ml-8">
                    <span className="text-purple-400">return</span>
                    <span className="text-gray-400"> n</span>
                  </div>
                  <div className="text-gray-300 ml-4">
                    <span className="text-purple-400">return</span>
                    <span className="text-gray-400"> calculate_fibonacci(n-</span>
                    <span className="text-orange-400">1</span>
                    <span className="text-gray-400">) + </span>
                  </div>
                  <div className="text-gray-300 ml-8">
                    <span className="text-gray-400">calculate_fibonacci(n-</span>
                    <span className="text-orange-400">2</span>
                    <span className="text-gray-400">)</span>
                  </div>
                  <div className="mt-4 text-gray-300">
                    <span className="text-gray-500"># AI {t('landing.hero.optimizedVersion')}</span>
                  </div>
                  <div className="text-green-400 mt-2">
                    <span>✓ {t('landing.hero.codeQuality')}</span>
                  </div>
                </div>
              </div>

              {/* Floating Badge */}
              <div className="absolute -bottom-4 -right-4 bg-white rounded-xl shadow-lg p-4 flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-green-100 flex items-center justify-center">
                  <Check className="w-5 h-5 text-green-600" />
                </div>
                <div>
                  <div className="text-sm font-semibold text-gray-900">{t('landing.hero.codeQuality')}</div>
                  <div className="text-xs text-gray-500">A+ {t('landing.hero.grade')}</div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.key}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                className="text-center"
              >
                <div className="text-3xl sm:text-4xl font-bold text-gray-900 mb-2">
                  {stat.value}
                </div>
                <div className="text-gray-600">{t(`landing.stats.${stat.key}`)}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
              {t('landing.features.title')}
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              {t('landing.features.subtitle')}
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.key}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                className="group p-6 bg-white rounded-2xl border border-gray-100 hover:border-gray-200 hover:shadow-lg transition-all"
              >
                <div
                  className="w-12 h-12 rounded-xl flex items-center justify-center mb-4 transition-transform group-hover:scale-110"
                  style={{ backgroundColor: `${feature.color}15` }}
                >
                  <feature.icon
                    className="w-6 h-6"
                    style={{ color: feature.color }}
                  />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  {t(`landing.features.${feature.key}.title`)}
                </h3>
                <p className="text-gray-600">{t(`landing.features.${feature.key}.description`)}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="py-20 bg-gray-50 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
              {t('landing.pricing.title')}
            </h2>
            <p className="text-lg text-gray-600">
              {t('landing.pricing.subtitle')}
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            {pricingPlans.map((plan, index) => (
              <motion.div
                key={plan.key}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                className={cn(
                  'relative p-8 rounded-2xl',
                  plan.highlighted
                    ? 'bg-gray-900 text-white shadow-xl scale-105'
                    : 'bg-white border border-gray-200'
                )}
              >
                {plan.highlighted && (
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2">
                    <span className="bg-[#FFD700] text-black text-sm font-semibold px-4 py-1 rounded-full">
                      {t('landing.pricing.pro.popular')}
                    </span>
                  </div>
                )}

                <h3
                  className={cn(
                    'text-xl font-semibold mb-2',
                    plan.highlighted ? 'text-white' : 'text-gray-900'
                  )}
                >
                  {t(`landing.pricing.${plan.key}.name`)}
                </h3>
                <div className="mb-4">
                  <span
                    className={cn(
                      'text-4xl font-bold',
                      plan.highlighted ? 'text-white' : 'text-gray-900'
                    )}
                  >
                    {plan.key === 'enterprise' ? t('landing.pricing.enterprise.custom') : `¥${plan.key === 'free' ? '0' : '99'}`}
                  </span>
                  <span
                    className={cn(
                      plan.highlighted ? 'text-gray-400' : 'text-gray-500'
                    )}
                  >
                    {plan.key !== 'enterprise' && '/月'}
                  </span>
                </div>
                <p
                  className={cn(
                    'mb-6',
                    plan.highlighted ? 'text-gray-400' : 'text-gray-600'
                  )}
                >
                  {t(`landing.pricing.${plan.key}.description`)}
                </p>

                <ul className="space-y-3 mb-8">
                  {[1, 2, 3, 4].map((i) => (
                    <li key={i} className="flex items-start gap-3">
                      <Check
                        className={cn(
                          'w-5 h-5 mt-0.5 flex-shrink-0',
                          plan.highlighted ? 'text-[#FFD700]' : 'text-green-500'
                        )}
                      />
                      <span
                        className={cn(
                          plan.highlighted ? 'text-gray-300' : 'text-gray-600'
                        )}
                      >
                        {t(`landing.pricing.${plan.key}.feature${i}`)}
                      </span>
                    </li>
                  ))}
                </ul>

                <RippleButton
                  className={cn(
                    'w-full',
                    plan.highlighted
                      ? 'bg-[#FFD700] text-black hover:bg-[#FFC700] font-semibold'
                      : 'bg-gray-100 text-gray-900 hover:bg-gray-200'
                  )}
                >
                  {t(`landing.pricing.${plan.key}.cta`)}
                </RippleButton>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
              {t('landing.cta.title')}
            </h2>
            <p className="text-lg text-gray-600 mb-8">
              {t('landing.cta.subtitle')}
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <RippleButton
                size="lg"
                className="bg-[#FFD700] text-black hover:bg-[#FFC700] font-semibold px-8"
              >
                {t('landing.cta.startFree')}
                <ArrowRight className="w-5 h-5 ml-2" />
              </RippleButton>
              <RippleButton
                size="lg"
                variant="outline"
                className="border-gray-300 text-gray-700 px-8"
              >
                {t('landing.cta.contactSales')}
              </RippleButton>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-[#0A1628] text-white py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-4 gap-12 mb-12">
            {/* Brand */}
            <div>
              <div className="flex items-center gap-2 mb-4">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#FFD700] to-[#FFA500] flex items-center justify-center">
                  <Code className="w-6 h-6 text-black" />
                </div>
                <span className="text-xl font-bold">CodeAI</span>
              </div>
              <p className="text-gray-400 mb-4">
                {t('landing.footer.tagline')}
              </p>
              <div className="flex gap-4">
                <a href="#" className="text-gray-400 hover:text-white transition-colors">
                  <Globe className="w-5 h-5" />
                </a>
                <a href="#" className="text-gray-400 hover:text-white transition-colors">
                  <Star className="w-5 h-5" />
                </a>
              </div>
            </div>

            {/* Links */}
            <div>
              <h4 className="font-semibold mb-4">{t('landing.footer.products')}</h4>
              <ul className="space-y-2">
                <li>
                  <a href="#" className="text-gray-400 hover:text-white transition-colors">
                    AI {t('landing.footer.codeGen')}
                  </a>
                </li>
                <li>
                  <a href="#" className="text-gray-400 hover:text-white transition-colors">
                    {t('landing.footer.cloudIDE')}
                  </a>
                </li>
                <li>
                  <a href="#" className="text-gray-400 hover:text-white transition-colors">
                    {t('landing.footer.smartDeploy')}
                  </a>
                </li>
                <li>
                  <a href="#" className="text-gray-400 hover:text-white transition-colors">
                    API {t('landing.footer.docs')}
                  </a>
                </li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold mb-4">{t('landing.footer.company')}</h4>
              <ul className="space-y-2">
                <li>
                  <a href="#" className="text-gray-400 hover:text-white transition-colors">
                    {t('landing.footer.about')}
                  </a>
                </li>
                <li>
                  <a href="#" className="text-gray-400 hover:text-white transition-colors">
                    {t('landing.footer.blog')}
                  </a>
                </li>
                <li>
                  <a href="#" className="text-gray-400 hover:text-white transition-colors">
                    {t('landing.footer.careers')}
                  </a>
                </li>
                <li>
                  <a href="#" className="text-gray-400 hover:text-white transition-colors">
                    {t('landing.footer.contact')}
                  </a>
                </li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold mb-4">{t('landing.footer.support')}</h4>
              <ul className="space-y-2">
                <li>
                  <a href="#" className="text-gray-400 hover:text-white transition-colors">
                    {t('landing.footer.helpCenter')}
                  </a>
                </li>
                <li>
                  <a href="#" className="text-gray-400 hover:text-white transition-colors">
                    {t('landing.footer.documentation')}
                  </a>
                </li>
                <li>
                  <a href="#" className="text-gray-400 hover:text-white transition-colors">
                    {t('landing.footer.community')}
                  </a>
                </li>
                <li>
                  <a href="#" className="text-gray-400 hover:text-white transition-colors">
                    {t('landing.footer.status')}
                  </a>
                </li>
              </ul>
            </div>
          </div>

          <div className="border-t border-gray-800 pt-8 flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-gray-400 text-sm">
              © 2024 CodeAI. All rights reserved.
            </p>
            <div className="flex gap-6 text-sm">
              <a href="#" className="text-gray-400 hover:text-white transition-colors">
                {t('landing.footer.privacy')}
              </a>
              <a href="#" className="text-gray-400 hover:text-white transition-colors">
                {t('landing.footer.terms')}
              </a>
              <a href="#" className="text-gray-400 hover:text-white transition-colors">
                {t('landing.footer.cookies')}
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
